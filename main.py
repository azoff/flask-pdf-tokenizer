from bs4 import BeautifulSoup
from fastapi import FastAPI, Response
from pdfminer.high_level import extract_text
from typing import Union
from urllib.request import urlretrieve
import hashlib
import logging
import os
import pdfkit
import pydantic
import re
import redis
import requests
import tempfile
import tiktoken

class DocsendRequest(pydantic.BaseModel):
  url: str
  email: str
  passcode: str = ''
  searchable: bool = True

class RenderRequest(pydantic.BaseModel):
  html: str

class ProxyRequest(pydantic.BaseModel):
  url: str
  method: str = 'GET'

class TextRequest(pydantic.BaseModel):
  url: str = ''
  extra_context: str = ''

class TruncateRequest(TextRequest):
  max_tokens: int = 2048
  model: str = "gpt-4"

logging.basicConfig(level=logging.INFO)

app = FastAPI()
cache = redis.Redis(host=os.getenv('REDIS_HOST'), port=6379, decode_responses=True)

@app.get("/")
def index():
  return {"status": "ok"}

@app.post("/text")
def text(req:TextRequest):
  text = download_pdf_and_extract_text(req.url, extra_context=req.extra_context)
  return { "text": text }

@app.post("/render")
def render(req:RenderRequest):
  html = req.html.encode('utf-8')
  digest = hashlib.sha256(html).hexdigest()
  headers = {'Content-Disposition': f'inline; filename="{digest}.pdf"', 'Content-Type': 'application/pdf'}
  pdf = pdf_from_html(html)
  resp = Response(content=pdf, headers=headers)
  return resp

@app.post("/docsend2pdf")
def docsend2pdf(req:DocsendRequest):
  headers, content = generate_pdf_from_docsend_url(
    req.url, 
    req.email, 
    passcode=req.passcode, 
    searchable=req.searchable
  )
  response = Response(content=content, headers=headers)
  return response

@app.post("/proxy")
def proxy(req:ProxyRequest):
  response = requests.request(req.method, req.url)
  return Response(content=response.content, headers=response.headers)

@app.post("/truncate")
def truncate(req:TruncateRequest):
  text = download_pdf_and_truncate_text(
    req.url, 
    extra_context=req.extra_context, 
    max_tokens=req.max_tokens,
    model=req.model
  )
  return { "text": text }

def docsend2pdf_credentials():
    # Make a GET request to fetch the initial page and extract CSRF tokens
    with requests.Session() as session:
        logging.info(f"Fetching docsend2pdf CSRF tokens...")
        response = session.get('https://docsend2pdf.com')
        if response.ok:
            cookies = session.cookies.get_dict()
            csrftoken = cookies.get('csrftoken', '')
            soup = BeautifulSoup(response.text, 'html.parser')
            csrfmiddlewaretoken = soup.find('input', {'name': 'csrfmiddlewaretoken'})['value']
            return {'csrfmiddlewaretoken': csrfmiddlewaretoken, 'csrftoken': csrftoken}
        else:
            response.raise_for_status()

def docsend2pdf_translate(url, csrfmiddlewaretoken, csrftoken, email, passcode='', searchable='on'):
    cache_key = f"docsend2pdf:{hash((url, email, passcode, searchable))}"
    if cache.exists(cache_key):
        logging.info(f"Using cache for {url}...")
        return cache.get(cache_key)
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
            'searchable': searchable
        }
        # Make a POST request to submit the form data
        logging.info(f"Converting {url} on behalf of {email}...")
        response = session.post('https://docsend2pdf.com', headers=headers, data=data, allow_redirects=True, timeout=60)
        if response.ok:
            logging.info(f"Conversion successful, received {response.headers['Content-Length']} bytes.")
            data = (response.headers, response.content)
            cache.set(cache_key, data)
            return data
        else:
            response.raise_for_status()

def generate_pdf_from_docsend_url(url, email, passcode='', searchable=True):
    credentials = docsend2pdf_credentials()
    kwargs = dict(
      email=email,
      passcode=passcode,
      searchable='on' if searchable else 'off',
      **credentials
    )
    return docsend2pdf_translate(url, **kwargs)


def download_pdf_and_truncate_text(url: str, extra_context: str = '', max_tokens:int = 2048, model:str = "gpt-4") -> str:
  text = download_pdf_and_extract_text(url, extra_context=extra_context)
  return truncate_text(text, max_tokens=max_tokens, model=model)

def download_pdf_and_extract_text(url: str, extra_context: str = '') -> str:
  
  if not url:
    logging.warning("No URL provided, returning only the input text.")
    return extra_context

  # hash the inputs into a cache key
  cache_key = f"text:{hash((url, extra_context))}"
  text = cache.get(cache_key)
  if (text is not None):
    logging.info(f"Using cache for {url}...")
    return text
  
  with tempfile.NamedTemporaryFile() as temp:
    download_pdf(url, temp.name)
    text = extract_text(temp.name)
    text = f"{text}{extra_context}"
  
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
