import base64
import hashlib
import io
import json
import logging
import os
import pickle
import re
from tempfile import (
    NamedTemporaryFile, TemporaryDirectory)
from typing import Any, Callable, TypeVar, Union

import docx
import pandas as pd
import pdf2image
import pdfkit
import pydantic
import PyPDF2
import pytesseract
import redis
import requests
import tiktoken
from docsend import DocSend
from fastapi import BackgroundTasks, FastAPI, Response
from pdfminer.high_level import extract_text
from PIL import Image
from requests import HTTPError

T = TypeVar('T', bound='BackgroundableRequest')

# fixes: https://stackoverflow.com/q/51152059
Image.MAX_IMAGE_PIXELS = 933120000

stripped_headers = ('transfer-encoding', 'content-encoding', 'content-length')


class ReferencableRequest(pydantic.BaseModel):
    reference: str = ''


class BackgroundableRequest(ReferencableRequest):
    asynchronous: bool = False


class DocsendRequest(BackgroundableRequest):
    url: str
    email: str
    passcode: str | None = None


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
cache = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'),
                    port=6379, decode_responses=True)


@app.exception_handler(HTTPError)
def http_exception_handler(_, exc: HTTPError):
    """
    catch requests.HTTPError and return the underlying status,
    and returning the error message in json
    """
    status_code = exc.response.status_code
    # use the exception error message as the error message
    error = str(exc)
    try:
        reason = exc.response.json()
    except json.JSONDecodeError:
        reason = exc.response.text
    content = json.dumps({'error': error, 'reason': reason})
    return Response(content=content, media_type='application/json', status_code=status_code)


# print errors as json
@app.exception_handler(Exception)
def uncaught_exception_handler(_, exc: Exception):
    return Response(content=json.dumps({'error': str(exc)}), media_type='application/json', status_code=500)


@app.get("/")
def index():
    return {"status": "ok"}


@app.post("/text")
def text(req: TextRequest):
    _text = download_pdf_and_extract_text(
        req.url, extra_context=req.extra_context)
    return {"text": _text}


@app.post("/render")
def render(req: RenderRequest):
    encoded = req.html.encode('utf-8')
    digest = hashlib.sha256(encoded).hexdigest()
    headers = {'Content-Disposition': f'inline; filename="{digest}.pdf"',
               'Content-Type': 'application/pdf'}
    pdf = pdf_from_html(req.html)
    kwargs = dict(content=pdf, headers=headers)
    return make_referenced_response(req.reference, kwargs)


@app.post("/docsend2pdf")
def docsend2pdf(req: DocsendRequest, background_tasks: BackgroundTasks):
    return background_request(req, background_tasks, docsend2pdf_sync)


@app.post("/proxy")
def proxy(req: ProxyRequest):
    kwargs = get_or_download_file(req.url, method=req.method)
    return make_referenced_response(req.reference, kwargs)


@app.post("/truncate")
def truncate(req: TruncateRequest, background_tasks: BackgroundTasks):
    return background_request(req, background_tasks, truncate_sync)


@app.post("/ocr")
def ocr(req: OCRRequest, background_tasks: BackgroundTasks):
    return background_request(req, background_tasks, ocr_sync)


@app.post("/xlsx2json")
def xlsx2json(req: OCRRequest, background_tasks: BackgroundTasks):
    return background_request(req, background_tasks, xlsx2json_sync)


@app.post("/docx2txt")
def docx2txt(req: OCRRequest, background_tasks: BackgroundTasks):
    return background_request(req, background_tasks, docx2txt_sync)


@app.get("/reference/{key}")
def reference(key: str):
    kwargs = get_reference_from_cache(key)
    if kwargs is None:
        logging.warning(f"Reference cache miss for {key}...")
        return Response(content=json.dumps({'error': 'Not Found', 'reason': f"{key} not found in reference cache."}),
                        media_type='application/json', status_code=404)
    ensure_content_downloadable(kwargs)
    return Response(**kwargs)


def background_request(req: T,
                       background_tasks: BackgroundTasks,
                       sync_handler: Callable[[T], Any]) -> dict[str, str]:
    resp: dict[str, str] = {}
    if req.asynchronous:
        background_tasks.add_task(sync_handler, req)
        resp = dict(key=make_reference_key(req.reference))
    else:
        resp = sync_handler(req)
    if 'key' in resp:
        logging.info(f"Returning reference key {resp['key']}...")
    return resp


def docsend2pdf_sync(req: DocsendRequest):
    kwargs = generate_pdf_from_docsend_url(
        req.url,
        req.email,
        passcode=req.passcode)
    return make_referenced_response(req.reference, kwargs)


def truncate_sync(req: TruncateRequest):
    _text = download_pdf_and_truncate_text(
        req.url,
        extra_context=req.extra_context,
        max_tokens=req.max_tokens,
        model=req.model
    )
    content = json.dumps({"text": _text})
    kwargs = dict(content=content, headers={
                  'Content-Type': 'application/json'})
    return make_referenced_response(req.reference, kwargs)


def get_reference_from_cache(key: str, default: Any = None) -> Any:
    if not cache.exists(key):
        return default
    return pickle.loads(base64.b64decode(str(cache.get(key))))


def extension_from_mimetype(mime_type: str) -> str:
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


def ensure_content_downloadable(kwargs: dict[str, Any]):
    if 'content' not in kwargs:
        raise ValueError("Missing content.")
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    if 'content' in kwargs['headers']:
        logging.warning("seems like a content header, seems like a mistake so I'm removing it...")
        del kwargs['headers']['content']
    if 'Content-Type' not in kwargs['headers']:
        raise ValueError("Missing content-type header.")
    if 'Content-Disposition' not in kwargs['headers']:
        # get extension from content-type
        content_type = kwargs.get('headers', dict()).get('Content-Type', '')
        extension = extension_from_mimetype(content_type)
        filename = f"{hashlib.sha256(kwargs['content']).hexdigest()}.{extension}"
        kwargs['headers']['Content-Disposition'] = f'inline; filename="{filename}"'


def make_referenced_response(seed: str, kwargs: dict[str, Any] | Response) -> Response | dict[str, str]:
    kwargs_dict = kwargs if isinstance(kwargs, dict) else dict(content=kwargs.body, headers=kwargs.headers)
    ensure_content_downloadable(kwargs_dict)
    if not seed:
        if isinstance(kwargs, dict):
            return Response(**kwargs)
        else:
            return kwargs
    key = make_reference_key(seed)
    logging.info(
        f"Caching {len(kwargs_dict['content'])} byte response under {key} for {seed}...")
    cache.set(key, base64.b64encode(pickle.dumps(kwargs_dict)))
    return dict(key=key)


def make_reference_key(seed: str) -> str:
    return hashlib.sha256(seed.encode('utf-8')).hexdigest()


def generate_pdf_from_docsend_url(url: str,
                                  email: str,
                                  passcode: str | None = None) -> dict[str, Any]:
    inputhash = hash((url, email, passcode))
    cache_key = f"docsend2pdf:{inputhash}"
    if cache.exists(cache_key):
        logging.info(f"Using cache for {url}...")
        return pickle.loads(base64.b64decode(str(cache.get(cache_key))))

    doc_id = url.split('/')[-1]
    doc = DocSend(doc_id=doc_id)
    logging.info(f"Attempting to download docsend doc {doc_id}...")

    logging.info("Fetching metadata...")
    doc.fetch_meta()  # type: ignore[attr-defined]

    assert doc.pages > 0, f"Docsend doc {doc_id} is empty!"
    logging.info(f"Docsend doc {doc_id} has {doc.pages} pages...")

    if email:
        logging.info(f"Authorizing using {email}...")
        if not passcode:
            passcode = None
        doc.authorize(email=email, passcode=passcode)  # type: ignore[attr-defined]

    logging.info("Fetching images...")
    doc.fetch_images()  # type: ignore[attr-defined]

    logging.info("Saving images to PDF...")
    with NamedTemporaryFile() as temp:
        doc.save_pdf(temp.name)  # type: ignore[attr-defined]
        with open(temp.name, 'rb') as f:
            content = f.read()
            headers = {
                'Content-Type': 'application/pdf',
                'Content-Disposition': f'inline; filename="{doc_id}.pdf"'
            }
            logging.info(
                f"Downloaded {url} to PDF, caching {len(content)} bytes under {cache_key}.")
            cache.set(cache_key, base64.b64encode(pickle.dumps(dict(content=content, headers=headers))))
            return dict(content=content, headers=headers)


def rasterize_pdf_to_images(pdf_path: str, output_folder: str | None = None):
    logging.info(f"Rasterizing PDF {pdf_path} to images...")
    images = pdf2image.convert_from_path(pdf_path, output_folder=output_folder)  # type: ignore[attr-defined]
    logging.info(f"Rasterized PDF to {len(images)} images.")
    return images


def ocr_image_to_pdf_page(image: Image.Image, i: int, total: int) -> PyPDF2.PageObject:
    logging.info(f"Running OCR on page {i+1} of {total}...")
    page = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')  # type: ignore[attr-defined]
    if isinstance(page, str):
        page = page.encode('utf-8')
    pdf = PyPDF2.PdfReader(io.BytesIO(page))
    return pdf.pages[0]


def convert_xlsx_to_json(url: str):
    cache_key = f"xlsx2json:{url}"
    if cache.exists(cache_key):
        logging.info(f"Using cache for {url}...")
        return pickle.loads(base64.b64decode(str(cache.get(cache_key))))

    with NamedTemporaryFile() as temp:
        get_or_download_file(url, temp)
        df = pd.read_excel(temp.name)  # type: ignore[attr-defined]
        json_data = df.to_json(orient='records')  # type: ignore[attr-defined]
        kwargs = dict(content=json_data.encode('utf-8'),
                      headers={'Content-Type': 'application/json'})
        logging.info(
            f"Converted {url} to JSON, caching {len(kwargs['content'])} bytes under {cache_key}.")
        cache.set(cache_key, base64.b64encode(pickle.dumps(kwargs)))
        return kwargs


def convert_docx_to_txt(url: str):
    cache_key = f"docx2txt:{url}"
    if cache.exists(cache_key):
        logging.info(f"Using cache for {url}...")
        return pickle.loads(base64.b64decode(str(cache.get(cache_key))))

    with NamedTemporaryFile() as temp:
        get_or_download_file(url, temp)
        with open(temp.name, 'rb') as f:
            doc = docx.Document(f)
            content = '\n'.join(
                [paragraph.text for paragraph in doc.paragraphs])
            kwargs = dict(content=content.encode('utf-8'),
                          headers={'Content-Type': 'text/plain'})
            logging.info(
                f"Converted {url} to text, caching {len(kwargs['content'])} bytes under {cache_key}.")
            cache.set(cache_key, base64.b64encode(pickle.dumps(kwargs)))
            return kwargs


def ocr_searchable_pdf(url: str) -> dict[str, Any]:
    cache_key = f"ocr_searchable_pdf:{url}"
    if cache.exists(cache_key):
        logging.info(f"Using cache for {url}...")
        return pickle.loads(base64.b64decode(str(cache.get(cache_key))))

    pages: list[PyPDF2.PageObject] = []
    with TemporaryDirectory() as raster_folder, NamedTemporaryFile() as temp:
        file_kwargs = get_or_download_file(url, temp)
        images = rasterize_pdf_to_images(temp.name, raster_folder)

        total = len(images)
        inputs = [(image, i, total) for i, image in enumerate(images)]
        for image, i, total in inputs:
            page = ocr_image_to_pdf_page(image, i, total)
            pages.append(page)  # type: ignore[attr-defined]
            # Free memory by closing the image
            image.close()
            del image

    pdf_writer = PyPDF2.PdfWriter()
    for page in pages:
        pdf_writer.add_page(page)

    # get the pdf bytes
    with io.BytesIO() as f:
        pdf_writer.write(f)  # type: ignore[attr-defined]
        f.seek(0)
        headers: dict[str, Any] = {'Content-Type': 'application/pdf'}
        file_headers: dict[str, Any] = file_kwargs.get('headers', {})  # type: ignore[attr-defined]
        for k, v in file_headers.items():  
            if k.lower() not in stripped_headers and v:
                headers[k] = str(v)
        kwargs = dict(content=f.read(), headers=headers)
        logging.info(
            f"OCR complete, caching {len(kwargs['content'])} bytes under {cache_key}.")
        cache.set(cache_key, base64.b64encode(pickle.dumps(kwargs)))
        return kwargs


def ocr_sync(req: OCRRequest):
    kwargs = ocr_searchable_pdf(req.url)
    return make_referenced_response(req.reference, kwargs)


def xlsx2json_sync(req: OCRRequest):
    kwargs = convert_xlsx_to_json(req.url)
    return make_referenced_response(req.reference, kwargs)


def docx2txt_sync(req: OCRRequest):
    kwargs = convert_docx_to_txt(req.url)
    return make_referenced_response(req.reference, kwargs)


def download_pdf_and_truncate_text(url: str,
                                   extra_context: str = '',
                                   max_tokens: int = 2048,
                                   model: str = "gpt-4") -> str:
    _text = download_pdf_and_extract_text(url, extra_context=extra_context)
    return truncate_text(_text, max_tokens=max_tokens, model=model)


def get_or_download_file(url: str,
                         temp: Any | None = None,
                         method: str = 'GET'):
    kwargs = None
    if url.startswith('ref:'):
        kwargs = get_reference_from_cache(url[4:])
    else:
        kwargs = download_file(url, method=method)
    if kwargs is None or kwargs.get('content') is None:
        raise ValueError(f"Unable to get or download {url}.")
    if temp is not None:
        content = kwargs['content']
        logging.info(
            f"Writing {len(content)} bytes to {temp.name}...")
        temp.write(content)
    return kwargs


def download_pdf_and_extract_text(url: str, extra_context: str = '') -> str:

    if not url:
        logging.warning("No URL provided, returning only the input text.")
        return extra_context

    # hash the inputs into a cache key
    cache_key = f"text:{hash((url, extra_context ))}"
    _text = cache.get(cache_key)
    if _text is not None:
        logging.info(f"Using cache for {url}...")
        return str(_text)

    with NamedTemporaryFile() as temp:
        get_or_download_file(url, temp)
        _text = extract_text(temp.name)

    # add extra text context
    # clean non-ascii characters from text
    # and remove redundant whitespace, plus trim
    _text_agg = f"{extra_context}\n\n{_text}"
    _text_agg = re.sub(r'[^\x00-\x7F]+', ' ', _text_agg)
    _text_agg = re.sub(r'\s+', ' ', _text_agg)
    _text_agg = _text_agg.strip()

    logging.info(
        f"Extracted {len(_text_agg)} characters from {url}, caching under {cache_key}.")
    cache.set(cache_key, _text_agg)
    return _text_agg


def clean_html(html: str) -> str:
    # Add proper HTML header with charset declaration
    if "<head>" not in html:
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Document</title>
</head>
<body>
{html}
</body>
</html>
"""
    else:
        # If there's already a head tag, just ensure charset is set
        if "charset" not in html:
            html = html.replace("<head>", '<head>\n    <meta charset="UTF-8">')
    # remove all image tags
    # see: https://github.com/wkhtmltopdf/wkhtmltopdf/issues/4408
    html = re.sub(r'<img[^>]*>', '', html)
    return html


def pdf_from_html(html: str, output_path: Union[str, bool] = False):
    config = pdfkit.configuration(wkhtmltopdf=os.getenv('WKHTMLTOPDF_PATH'))  # type: ignore[attr-defined]
    options = {
        "load-error-handling": "ignore",
        "load-media-error-handling": "ignore",
        "encoding": "UTF-8",
        "enable-local-file-access": "",
        "disable-smart-shrinking": "",
        "no-background": "",
    }
    html = clean_html(html)
    return pdfkit.from_string(html, output_path, options=options, configuration=config)  # type: ignore[attr-defined]


def download_file(url: str, method: str = 'GET'):
    logging.info(f"Downloading: {method} {url}...")
    response = requests.request(method, url, timeout=60)
    if response.ok:
        logging.info(f"Downloaded {len(response.content)} bytes from {url}.")
        headers = {k: v for k, v in response.headers.items() if k.lower()
                   not in stripped_headers}
        kwargs = dict(content=response.content, headers=headers)
        return kwargs
    else:
        message = f"Failed to download {url}: {response.status_code} {response.reason}"
        logging.error(message)
        response.status_code = 502  # Bad Gateway
        raise requests.HTTPError(message, response=response)


def truncate_text(_text: str, max_tokens: int = 2048, model: str = "gpt-4") -> str:
    _text = _text.replace('\n', ' ')
    _text = re.sub(r'[\s\W]([\S\w][\s\W])+', ' ', _text)
    _text = re.sub(r'\s+', ' ', _text)
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(_text)
    trim = max_tokens - len(tokens)
    if trim >= 0:
        logging.info(f"No truncation needed, {len(tokens)} <= {max_tokens}.")
        return _text

    logging.info(f"Truncating {len(tokens)} to {max_tokens}...")
    return encoding.decode(tokens[:trim])
