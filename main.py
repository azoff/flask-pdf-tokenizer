from bs4 import BeautifulSoup
from fastapi import FastAPI, Response
from pdfminer.high_level import extract_text
from typing import Union
from urllib.request import urlretrieve
import base64
import gzip
import hashlib
import logging
import os
import pdf2image
import pickle
import pdfkit
import pydantic
import pytesseract
import re
import redis
import requests
import tempfile
import tiktoken
import time

class ReferencableRequest(pydantic.BaseModel):
   reference: str = ''

class DocsendRequest(ReferencableRequest):
  url: str
  email: str
  passcode: str = ''
  searchable: bool = True

class RenderRequest(ReferencableRequest):
  html: str

class ProxyRequest(ReferencableRequest):
  url: str
  method: str = 'GET'

class TextRequest(pydantic.BaseModel):
  url: str = ''
  extra_context: str = ''
  ocr: bool = False

class TruncateRequest(TextRequest):
  max_tokens: int = 2048
  model: str = "gpt-4"

class OCRRequest(pydantic.BaseModel):
  url: str
  extra_context: str = ''

logging.basicConfig(level=logging.INFO)

app = FastAPI()
cache = redis.Redis(host=os.getenv('REDIS_HOST'), port=6379, decode_responses=True)

@app.get("/")
def index():
  return {"status": "ok"}

@app.post("/text")
def text(req:TextRequest):
  text = download_pdf_and_extract_text(req.url, extra_context=req.extra_context, ocr=req.ocr)
  return { "text": text }

@app.post("/render")
def render(req:RenderRequest):
  html = req.html.encode('utf-8')
  digest = hashlib.sha256(html).hexdigest()
  headers = {'Content-Disposition': f'inline; filename="{digest}.pdf"', 'Content-Type': 'application/pdf'}
  pdf = pdf_from_html(html)
  kwargs = dict(content=pdf, headers=headers)
  return make_referenced_response(req.reference, kwargs)

@app.post("/docsend2pdf")
def docsend2pdf(req:DocsendRequest):
  kwargs = generate_pdf_from_docsend_url(
    req.url, 
    req.email, 
    passcode=req.passcode, 
    searchable=req.searchable
  )
  return make_referenced_response(req.reference, kwargs)

@app.post("/proxy")
def proxy(req:ProxyRequest):
  response = requests.request(req.method, req.url)
  kwargs = dict(content=response.content, headers=response.headers)
  return make_referenced_response(req.reference, kwargs)

@app.post("/truncate")
def truncate(req:TruncateRequest):
  text = download_pdf_and_truncate_text(
    req.url, 
    extra_context=req.extra_context,
    ocr=req.ocr,
    max_tokens=req.max_tokens,
    model=req.model
  )
  return { "text": text }

@app.post("/ocr")
def ocr(req:OCRRequest):
  text = ocr_remote_pdf(req.url)
  text = f"{text} {req.extra_context}"
  return { "text": text.strip() }

@app.get("/reference/{key}")
def reference(key:str):
  kwargs = pickle.loads(base64.b64decode(cache.get(key)))
  return Response(**kwargs)

def make_referenced_response(seed, kwargs):
  if not seed:
     return Response(**kwargs)
  key = hashlib.sha256(seed.encode('utf-8')).hexdigest()
  cache.set(key, base64.b64encode(pickle.dumps(kwargs)))
  return dict(key=key)

def docsend2pdf_credentials():
    # Make a GET request to fetch the initial page and extract CSRF tokens
    with requests.Session() as session:
        start_time = time.time()
        logging.info(f"Fetching docsend2pdf CSRF tokens...")
        response = session.get('https://docsend2pdf.com')
        logging.info(f"Received docsend2pdf CSRF tokens in {time.time() - start_time} seconds.")
        if response.ok:
            cookies = session.cookies.get_dict()
            csrftoken = cookies.get('csrftoken', '')
            soup = BeautifulSoup(response.text, 'html.parser')
            csrfmiddlewaretoken = soup.find('input', {'name': 'csrfmiddlewaretoken'})['value']
            return {'csrfmiddlewaretoken': csrfmiddlewaretoken, 'csrftoken': csrftoken}
        else:
            response.raise_for_status()

def docsend2pdf_translate(url, csrfmiddlewaretoken, csrftoken, email, passcode='', searchable=False):
    inputhash = hash((url, email, passcode, searchable))
    cache_key = f"docsend2pdf:{inputhash}"
    if cache.exists(cache_key):
        logging.info(f"Using cache for {url}...")
        return pickle.loads(base64.b64decode(cache.get(cache_key)))
    with requests.Session() as session:
        # Include csrftoken in session cookies
        session.cookies.set('csrftoken', csrftoken)
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Referer': 'https://docsend2pdf.com/'
        }
        data = {
            'csrfmiddlewaretoken': csrfmiddlewaretoken,
            'url': url,
            'email': email,
            'passcode': passcode,
        }
        if searchable:
            data['searchable'] = 'on'
        # Make a POST request to submit the form data
        start_time = time.time()
        logging.info(f"Converting {url} on behalf of {email}...")
        response = session.post('https://docsend2pdf.com', headers=headers, data=data, allow_redirects=True, timeout=60)
        if response.ok:
            logging.info(f"Conversion successful, received {response.headers['Content-Length']} bytes in {time.time() - start_time} seconds.")
            # gzip content
            kwargs = dict(
              content=response.content,
              headers={
                'Content-Type': response.headers['Content-Type'],
                'Content-Disposition': response.headers.get('Content-Disposition', f'inline; filename="{inputhash}.pdf"')
              }
            )
            cache.set(cache_key, base64.b64encode(pickle.dumps(kwargs)))
            return kwargs
        else:
            response.raise_for_status()

def rasterize_pdf_to_images(pdf_bytes):
    return pdf2image.convert_from_bytes(pdf_bytes)

def ocr_image(image):
    return pytesseract.image_to_string(image)

def ocr_pdf_bytes(pdf_bytes):
    images = rasterize_pdf_to_images(pdf_bytes)
    return ' '.join([ocr_image(image) for image in images])

def ocr_remote_pdf(url):
    response = requests.get(url)
    return ocr_pdf_bytes(response.content)

def generate_pdf_from_docsend_url(url, email, passcode='', searchable=True):
    credentials = docsend2pdf_credentials()
    kwargs = dict(
      email=email,
      passcode=passcode,
      searchable=searchable,
      **credentials
    )
    return docsend2pdf_translate(url, **kwargs)

def download_pdf_and_truncate_text(url: str, extra_context: str = '', ocr:bool = False, max_tokens:int = 2048, model:str = "gpt-4") -> str:
  text = download_pdf_and_extract_text(url, extra_context=extra_context, ocr=ocr)
  return truncate_text(text, max_tokens=max_tokens, model=model)

def download_pdf_and_extract_text(url: str, extra_context: str = '', ocr:bool = False) -> str:
  
  if not url:
    logging.warning("No URL provided, returning only the input text.")
    return extra_context

  # hash the inputs into a cache key
  cache_key = f"text:{hash((url, extra_context, ocr))}"
  text = cache.get(cache_key)
  if (text is not None):
    logging.info(f"Using cache for {url}...")
    return text
  
  with tempfile.NamedTemporaryFile() as temp:
    download_pdf(url, temp.name)
    text = extract_text(temp.name)
    if ocr:
      text = f"{text} {ocr_pdf_bytes(temp.read())}"
    text = f"{text} {extra_context}".strip()
  
  cache.set(cache_key, text)
  return text

def pdf_from_html(html:str, output_path: Union[str, bool] = False):
  config = pdfkit.configuration(wkhtmltopdf=os.getenv('WKHTMLTOPDF_PATH'))
  options = {"load-error-handling": "ignore", "load-media-error-handling": "ignore"}
  # remove all image tags
  # see: https://github.com/wkhtmltopdf/wkhtmltopdf/issues/4408
  html = re.sub(r'<img[^>]*>', '', html)
  return pdfkit.from_string(html, output_path, options=options, configuration=config)

def download_pdf(url, output_path):
  logging.info(f"Downloading PDF from {url}...")
  pdf = urlretrieve(url, output_path)
  logging.info(f"PDF downloaded.")
  return pdf

def truncate_text(text:str, max_tokens:int = 2048, model:str = "gpt-4") -> str:
  text = text.replace('\n', ' ')
  text = re.sub('[\s\W]([\S\w][\s\W])+', ' ', text)
  text = re.sub('\s+', ' ', text)
  encoding = tiktoken.encoding_for_model(model)
  tokens = encoding.encode(text)
  trim = max_tokens - len(tokens)
  if trim >= 0:
    logging.info(f"No truncation needed, {len(tokens)} <= {max_tokens}.")
    return text
  
  logging.info(f"Truncating {len(tokens)} to {max_tokens}...")
  return encoding.decode(tokens[:trim])
