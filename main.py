from bs4 import BeautifulSoup
from fastapi import FastAPI, Response, BackgroundTasks
from pdfminer.high_level import extract_text
from typing import Union
from urllib.request import urlretrieve
import base64
import hashlib
import io
import json
import docx
import logging
import os
import pandas as pd
import pdf2image
import pdfkit
import pickle
import pydantic
import PyPDF2
import pytesseract
import re
import redis
import requests
import tempfile
import tiktoken
import time
import PIL

# because: https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
PIL.Image.MAX_IMAGE_PIXELS = 933120000

stripped_headers = ('transfer-encoding', 'content-encoding', 'content-length')

class ReferencableRequest(pydantic.BaseModel):
  reference: str = ''

class BackgroundableRequest(ReferencableRequest):
  asynchronous: bool = False

class DocsendRequest(BackgroundableRequest):
  url: str
  email: str
  passcode: str = ''
  searchable: bool = True

class RenderRequest(ReferencableRequest):
  html: str

class ProxyRequest(ReferencableRequest):
  url: str
  method: str = 'GET'

class TextRequest(BackgroundableRequest):
  url: str = ''
  extra_context: str = ''

class TruncateRequest(TextRequest):
  max_tokens: int = 2048
  model: str = "gpt-4"

class OCRRequest(BackgroundableRequest):
  url: str

logging.basicConfig(level=logging.INFO)

app = FastAPI()
cache = redis.Redis(host=os.getenv('REDIS_HOST'), port=6379, decode_responses=True)

# catch requests.HTTPError and return the underlying status, and returning the error message in json
@app.exception_handler(requests.HTTPError)
def http_exception_handler(request, exc):
  status_code = exc.response.status_code
  # use the exception error message as the error message
  if hasattr(exc, 'message'):
    error = exc.message
  else:
    error = str(exc)
  try:
    reason = exc.response.json()
  except json.JSONDecodeError:
    reason = exc.response.text
  content = json.dumps({ 'error': error, 'reason': reason })
  return Response(content=content, media_type='application/json', status_code=status_code)


# print errors as json
@app.exception_handler(Exception)
def uncaught_exception_handler(_, exc):
  return Response(content=json.dumps({ 'error': str(exc) }), media_type='application/json', status_code=500)

@app.get("/")
def index():
  return {"status": "ok"}

@app.post("/text")
def text(req:TextRequest):
  text = download_pdf_and_extract_text(req.url, extra_context=req.extra_context)
  return { "text": text }

@app.post("/render")
def render(req:RenderRequest):
  encoded = req.html.encode('utf-8')
  digest = hashlib.sha256(encoded).hexdigest()
  headers = {'Content-Disposition': f'inline; filename="{digest}.pdf"', 'Content-Type': 'application/pdf'}
  pdf = pdf_from_html(req.html)
  kwargs = dict(content=pdf, headers=headers)
  return make_referenced_response(req.reference, kwargs)

@app.post("/docsend2pdf")
def docsend2pdf(req:DocsendRequest, background_tasks: BackgroundTasks):
  return background_request(req, background_tasks, docsend2pdf_sync)


@app.post("/proxy")
def proxy(req:ProxyRequest):
  kwargs = get_or_download_file(req.url, method=req.method)
  return make_referenced_response(req.reference, kwargs)

@app.post("/truncate")
def truncate(req:TruncateRequest, background_tasks: BackgroundTasks):
  return background_request(req, background_tasks, truncate_sync)
  
@app.post("/ocr")
def ocr(req:OCRRequest, background_tasks: BackgroundTasks):
  return background_request(req, background_tasks, ocr_sync)

@app.post("/xlsx2json")
def xlsx2json(req:OCRRequest, background_tasks: BackgroundTasks):
  return background_request(req, background_tasks, xlsx2json_sync)

@app.post("/docx2txt")
def docx2txt(req:OCRRequest, background_tasks: BackgroundTasks):
  return background_request(req, background_tasks, docx2txt_sync)

@app.get("/reference/{key}")
def reference(key:str):
  kwargs = get_reference_from_cache(key)
  ensure_content_downloadable(kwargs)
  return Response(**kwargs)

def background_request(req:BackgroundableRequest, background_tasks: BackgroundTasks, sync_handler: callable):
  resp = None
  if req.asynchronous:
    background_tasks.add_task(sync_handler, req)
    resp = dict(key=make_reference_key(req.reference))
  else:
    resp = sync_handler(req)
  if isinstance(resp, dict) and 'key' in resp:
    logging.info(f"Returning reference key {resp['key']}...")
  return resp

def docsend2pdf_sync(req:DocsendRequest):
   kwargs = generate_pdf_from_docsend_url(
    req.url, 
    req.email, 
    passcode=req.passcode, 
    searchable=req.searchable)
   return make_referenced_response(req.reference, kwargs)

def truncate_sync(req:TruncateRequest):
  text = download_pdf_and_truncate_text(
    req.url, 
    extra_context=req.extra_context,
    max_tokens=req.max_tokens,
    model=req.model
  )
  content = json.dumps({ "text": text })
  kwargs = dict(content=content, headers={'Content-Type': 'application/json'})
  return make_referenced_response(req.reference, kwargs)

def get_reference_from_cache(key, default=None):
  if not cache.exists(key):
    return default
  return pickle.loads(base64.b64decode(cache.get(key)))

def extension_from_mimetype(mime_type):
  mime_types = {
    'application/pdf': 'pdf',
    'application/msword': 'docx',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'application/vnd.ms-excel': 'xls',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
    'text/plain': 'txt',
    'text/html': 'html',
    'text/csv': 'csv',
    'application/json': 'json',
    'image/jpeg': 'jpg',
    'image/png': 'png',
    'image/gif': 'gif',
  }
  return mime_types.get(mime_type, 'blob')

def ensure_content_downloadable(kwargs):
  if 'content' not in kwargs:
    raise ValueError("Missing content.")
  if 'headers' not in kwargs:
    kwargs['headers'] = {}
  if 'Content-Type' not in kwargs['headers']:
    raise ValueError("Missing content-type header.")
  if 'Content-Disposition' not in kwargs['headers']:
    # get extension from content-type
    content_type = kwargs['headers']['Content-Type']
    extension = extension_from_mimetype(content_type)
    filename = f"{hashlib.sha256(kwargs['content']).hexdigest()}.{extension}"
    kwargs['headers']['Content-Disposition'] = f'inline; filename="{filename}"'

def make_referenced_response(seed, kwargs):
  ensure_content_downloadable(kwargs)
  if not seed:
     return Response(**kwargs)
  key = make_reference_key(seed)
  logging.info(f"Caching {len(kwargs['content'])} byte response under {key} for {seed}...")
  cache.set(key, base64.b64encode(pickle.dumps(kwargs)))
  return dict(key=key)

def make_reference_key(seed):
  return hashlib.sha256(seed.encode('utf-8')).hexdigest()

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

def rasterize_pdf_to_images(pdf_path:str, output_folder:str = None):
    logging.info(f"Rasterizing PDF {pdf_path} to images...")
    images = pdf2image.convert_from_path(pdf_path, output_folder=output_folder)
    logging.info(f"Rasterized PDF to {len(images)} images.")
    return images

def ocr_image_to_pdf_page(image, i, total):
    logging.info(f"Running OCR on page {i+1} of {total}...")
    page = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')
    pdf = PyPDF2.PdfReader(io.BytesIO(page))
    return pdf.pages[0]

def convert_xlsx_to_json(url):
  cache_key = f"xlsx2json:{url}"
  if cache.exists(cache_key):
    logging.info(f"Using cache for {url}...")
    return pickle.loads(base64.b64decode(cache.get(cache_key)))
  
  with tempfile.NamedTemporaryFile() as temp:
    get_or_download_file(url, temp)
    df = pd.read_excel(temp.name)
    json_data = df.to_json(orient='records')
    kwargs = dict(content=json_data.encode('utf-8'), headers={'Content-Type': 'application/json'})
    logging.info(f"Converted {url} to JSON, caching {len(kwargs['content'])} bytes under {cache_key}.")
    cache.set(cache_key, base64.b64encode(pickle.dumps(kwargs)))
    return kwargs
  
def convert_docx_to_txt(url):
  cache_key = f"docx2txt:{url}"
  if cache.exists(cache_key):
    logging.info(f"Using cache for {url}...")
    return pickle.loads(base64.b64decode(cache.get(cache_key)))
  
  with tempfile.NamedTemporaryFile() as temp:
    get_or_download_file(url, temp)
    with open(temp.name, 'rb') as f:
      doc = docx.Document(f)
      content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
      kwargs = dict(content=content.encode('utf-8'), headers={'Content-Type': 'text/plain'})
      logging.info(f"Converted {url} to text, caching {len(kwargs['content'])} bytes under {cache_key}.")
      cache.set(cache_key, base64.b64encode(pickle.dumps(kwargs)))
      return kwargs

def ocr_searchable_pdf(url):
  cache_key = f"ocr_searchable_pdf:{url}"
  if cache.exists(cache_key):
    logging.info(f"Using cache for {url}...")
    return pickle.loads(base64.b64decode(cache.get(cache_key)))
  
  pages = []
  with tempfile.TemporaryDirectory() as raster_folder, tempfile.NamedTemporaryFile() as temp:
    file_kwargs = get_or_download_file(url, temp)
    images = rasterize_pdf_to_images(temp.name, raster_folder)

    total = len(images)
    inputs = [(image, i, total) for i, image in enumerate(images)]    
    for image, i, total in inputs:
      page = ocr_image_to_pdf_page(image, i, total)
      pages.append(page)
      # Free memory by closing the image
      image.close()
      del image
  
  pdf_writer = PyPDF2.PdfWriter()
  for page in pages:
    pdf_writer.add_page(page)
  
  # get the pdf bytes
  with io.BytesIO() as f:
    pdf_writer.write(f)
    f.seek(0)
    headers = { k: v for k, v in file_kwargs['headers'].items() if k.lower() not in stripped_headers}
    kwargs = dict(content=f.read(), headers=headers)
    logging.info(f"OCR complete, caching {len(kwargs['content'])} bytes under {cache_key}.")
    cache.set(cache_key, base64.b64encode(pickle.dumps(kwargs)))
    return kwargs

def ocr_sync(req:OCRRequest):
  kwargs = ocr_searchable_pdf(req.url)
  return make_referenced_response(req.reference, kwargs)

def xlsx2json_sync(req:OCRRequest):
  kwargs = convert_xlsx_to_json(req.url)
  return make_referenced_response(req.reference, kwargs)

def docx2txt_sync(req:OCRRequest):
  kwargs = convert_docx_to_txt(req.url)
  return make_referenced_response(req.reference, kwargs)

def generate_pdf_from_docsend_url(url, email, passcode='', searchable=True):
    credentials = docsend2pdf_credentials()
    kwargs = dict(
      email=email,
      passcode=passcode,
      searchable=searchable,
      **credentials
    )
    return docsend2pdf_translate(url, **kwargs)

def download_pdf_and_truncate_text(url: str, extra_context: str = '', max_tokens:int = 2048, model:str = "gpt-4") -> str:
  text = download_pdf_and_extract_text(url, extra_context=extra_context)
  return truncate_text(text, max_tokens=max_tokens, model=model)

def get_or_download_file(url: str, temp: tempfile.NamedTemporaryFile = None, method = 'GET'):
  kwargs = None
  if url.startswith('ref:'):
    kwargs = get_reference_from_cache(url[4:])
  else:
    kwargs = download_file(url, method=method)
  if kwargs is None or kwargs.get('content') is None:
      raise ValueError(f"Unable to get or download {url}.")
  if temp is not None:
    logging.info(f"Writing {len(kwargs['content'])} bytes to {temp.name}...")
    temp.write(kwargs['content'])
  return kwargs

def download_pdf_and_extract_text(url: str, extra_context: str = '') -> str:
  
  if not url:
    logging.warning("No URL provided, returning only the input text.")
    return extra_context

  # hash the inputs into a cache key
  cache_key = f"text:{hash((url, extra_context ))}"
  text = cache.get(cache_key)
  if (text is not None):
    logging.info(f"Using cache for {url}...")
    return text

  with tempfile.NamedTemporaryFile() as temp:  
    get_or_download_file(url, temp)
    text = extract_text(temp.name)

  # add extra text context
  # clean non-ascii characters from text
  # and remove redundant whitespace, plus trim
  text = f"{extra_context}\n\n{text}"
  text = re.sub(r'[^\x00-\x7F]+', ' ', text)
  text = re.sub(r'\s+', ' ', text)
  text = text.strip()

  logging.info(f"Extracted {len(text)} characters from {url}, caching under {cache_key}.")    
  cache.set(cache_key, text)
  return text

def pdf_from_html(html:str, output_path: Union[str, bool] = False):
  config = pdfkit.configuration(wkhtmltopdf=os.getenv('WKHTMLTOPDF_PATH'))
  options = {"load-error-handling": "ignore", "load-media-error-handling": "ignore"}
  # remove all image tags
  # see: https://github.com/wkhtmltopdf/wkhtmltopdf/issues/4408
  html = re.sub(r'<img[^>]*>', '', html)
  return pdfkit.from_string(html, output_path, options=options, configuration=config)

def download_file(url, method='GET'):
  logging.info(f"Downloading: {method} {url}...")
  response = requests.request(method, url)
  if response.ok:
    logging.info(f"Downloaded {len(response.content)} bytes from {url}.")
    headers = {k: v for k, v in response.headers.items() if k.lower() not in stripped_headers}
    kwargs = dict(content=response.content, headers=headers)
    return kwargs
  else:
    message = f"Failed to download {url}: {response.status_code} {response.reason}"
    logging.error(message)
    response.status_code = 502 # Bad Gateway
    raise requests.HTTPError(message, response=response)

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
