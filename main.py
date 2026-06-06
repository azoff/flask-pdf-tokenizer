import base64
import hashlib
import io
import json
import logging
import os
import pickle
import re
import secrets
from tempfile import (
    NamedTemporaryFile, TemporaryDirectory)
from typing import Any, Callable, TypeVar, Union
from urllib.parse import urlparse
from typing import cast

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
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pdfminer.high_level import extract_text
from PIL import Image
from requests import HTTPError
import subprocess
import shutil

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


class PptxRequest(BackgroundableRequest):
    url: str


logging.basicConfig(level=logging.INFO)

app = FastAPI()
cache = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'),
                    port=6379, decode_responses=True)

write_auth_scheme = HTTPBasic(auto_error=False)


def require_write_auth(
    credentials: HTTPBasicCredentials | None = Depends(write_auth_scheme),
):
    expected = os.getenv('HTTP_WRITE_AUTH', '')
    if not expected:
        return
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Authentication required.',
            headers={'WWW-Authenticate': 'Basic'},
        )
    expected_user, _, expected_pass = expected.partition(':')
    user_ok = secrets.compare_digest(
        credentials.username.encode('utf-8'),
        expected_user.encode('utf-8'),
    )
    pass_ok = secrets.compare_digest(
        credentials.password.encode('utf-8'),
        expected_pass.encode('utf-8'),
    )
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Invalid credentials.',
        )


write_auth = [Depends(require_write_auth)]


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


@app.post("/text", dependencies=write_auth)
def text(req: TextRequest):
    _text = download_pdf_and_extract_text(
        req.url, extra_context=req.extra_context)
    return {"text": _text}


@app.post("/render", dependencies=write_auth)
def render(req: RenderRequest):
    encoded = req.html.encode('utf-8')
    digest = hashlib.sha256(encoded).hexdigest()
    headers = {'Content-Disposition': f'inline; filename="{digest}.pdf"',
               'Content-Type': 'application/pdf'}
    pdf = pdf_from_html(req.html)
    kwargs = dict(content=pdf, headers=headers)
    return make_referenced_response(req.reference, kwargs)


@app.post("/docsend2pdf", dependencies=write_auth)
def docsend2pdf(req: DocsendRequest, background_tasks: BackgroundTasks):
    return background_request(req, background_tasks, docsend2pdf_sync)


@app.post("/proxy", dependencies=write_auth)
def proxy(req: ProxyRequest):
    kwargs = get_or_download_file(req.url, method=req.method)
    return make_referenced_response(req.reference, kwargs)


@app.post("/truncate", dependencies=write_auth)
def truncate(req: TruncateRequest, background_tasks: BackgroundTasks):
    return background_request(req, background_tasks, truncate_sync)


@app.post("/ocr", dependencies=write_auth)
def ocr(req: OCRRequest, background_tasks: BackgroundTasks):
    return background_request(req, background_tasks, ocr_sync)


@app.post("/pptx2pdf", dependencies=write_auth)
def pptx2pdf(req: PptxRequest, background_tasks: BackgroundTasks):
    return background_request(req, background_tasks, pptx2pdf_sync)


@app.post("/xlsx2json", dependencies=write_auth)
def xlsx2json(req: OCRRequest, background_tasks: BackgroundTasks):
    return background_request(req, background_tasks, xlsx2json_sync)


@app.post("/docx2txt", dependencies=write_auth)
def docx2txt(req: OCRRequest, background_tasks: BackgroundTasks):
    return background_request(req, background_tasks, docx2txt_sync)


@app.post("/upload", dependencies=write_auth)
async def upload(request: Request, reference: str = ''):
    """Ingest raw file bytes (request body) into the reference cache.

    Companion to /proxy for callers that hold a local file rather than a URL.
    POST the file bytes as the body with a `?reference=<seed>` query param and
    a Content-Type header; returns {"key": ...} usable as `ref:<key>` by the
    extraction endpoints (/ocr, /docx2txt, /xlsx2json, /pptx2pdf, /text).
    """
    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="empty upload body")
    content_type = request.headers.get('content-type') or 'application/octet-stream'
    return make_referenced_response(reference, dict(content=content, headers={'Content-Type': content_type}))


@app.get("/reference/{key}")
def reference(key: str):
    kwargs = get_reference_from_cache(key)
    if kwargs is None:
        logging.warning(f"Reference cache miss for {key}...")
        return Response(content=json.dumps({'error': 'Not Found', 'reason': f"{key} not found in reference cache."}),
                        media_type='application/json', status_code=404)
    ensure_content_downloadable(kwargs)
    return Response(**kwargs)


# Single-flight lock: heavy synchronous ops (ocr/docsend2pdf/pptx2pdf/etc.)
# run one at a time per container, so concurrent jobs can't saturate CPU/RAM.
# A busy request gets 429 + Retry-After; the caller should re-enqueue. The TTL
# is a safety release in case a holder crashes mid-op.
HEAVY_LOCK_KEY = "pdf:heavy-lock"
HEAVY_LOCK_TTL = int(os.getenv("PDF_HEAVY_LOCK_TTL", "900"))


def background_request(req: T,
                       background_tasks: BackgroundTasks,
                       sync_handler: Callable[[T], Any]) -> dict[str, str] | object:
    if req.asynchronous:
        background_tasks.add_task(sync_handler, req)
        reference_key = make_reference_key(req.reference)
        logging.info(f"Returning async reference key {reference_key}...")
        return {'key': reference_key}

    if not cache.set(HEAVY_LOCK_KEY, "1", nx=True, ex=HEAVY_LOCK_TTL):
        logging.info("Heavy op already running — returning 429 busy.")
        raise HTTPException(status_code=429, detail="pdf service busy with another heavy job",
                            headers={"Retry-After": "20"})
    try:
        resp: object = sync_handler(req)
        response_type = str(type(resp))
        logging.info(f"Returning synchronous response {response_type}...")
        return resp
    finally:
        cache.delete(HEAVY_LOCK_KEY)


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
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
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

    content_type = kwargs.get('headers', {}).get('Content-Type', 'application/octet-stream')
    if 'Content-Disposition' not in kwargs['headers']:
        # get extension from content-type
        extension = extension_from_mimetype(content_type)
        filename = f"{hashlib.sha256(kwargs['content']).hexdigest()}.{extension}"
        kwargs['headers']['Content-Disposition'] = f'inline; filename="{filename}"'

    # If content-type is application/octet-stream, try to infer from filename extension
    if content_type == 'application/octet-stream':
        content_disposition = kwargs.get('headers', {}).get('Content-Disposition', '')
        # Extract filename from Content-Disposition header
        filename_match = re.search(r'filename="([^"]+)"', content_disposition)
        if filename_match:
            filename = filename_match.group(1)
            # Get extension from filename
            if '.' in filename:
                extension = filename.split('.')[-1].lower()
                # Map extension to MIME type
                extension_to_mime = {
                    'pdf': 'application/pdf',
                    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                    'txt': 'text/plain',
                    'html': 'text/html',
                    'csv': 'text/csv',
                    'json': 'application/json',
                    'jpg': 'image/jpeg',
                    'jpeg': 'image/jpeg',
                    'png': 'image/png',
                    'gif': 'image/gif',
                }
                if extension in extension_to_mime:
                    kwargs['headers']['Content-Type'] = extension_to_mime[extension]
                    logging.info(
                        "Updated Content-Type from application/octet-stream to %s "
                        "based on filename extension .%s",
                        extension_to_mime[extension],
                        extension,
                    )


# --- PPTX to PDF support ---

def _is_pptx(headers: dict[str, Any]) -> tuple[bool, str]:
    """Return (is_pptx, suggested_basename) based on headers."""
    ctype = headers.get('Content-Type', '')
    if 'application/vnd.openxmlformats-officedocument.presentationml.presentation' in ctype:
        # try to pull filename stem from content-disposition
        cd = headers.get('Content-Disposition', '')
        m = re.search(r'filename="([^"]+)"', cd)
        name = m.group(1) if m else 'presentation'
        stem = os.path.splitext(name)[0]
        return True, stem
    # Try by filename extension if available
    cd = headers.get('Content-Disposition', '')
    m = re.search(r'filename="([^"]+)"', cd)
    if m:
        name = m.group(1)
        if name.lower().endswith('.pptx'):
            return True, os.path.splitext(name)[0]
    return False, ''


def _convert_pptx_bytes_to_pdf(pptx_bytes: bytes, basename: str | None = None) -> bytes:
    soffice = shutil.which('soffice') or shutil.which('libreoffice')
    if not soffice:
        raise RuntimeError(
            'LibreOffice (soffice) not found. Please install libreoffice '
            'to enable PPTX->PDF conversion.'
        )
    with TemporaryDirectory() as td:
        in_path = os.path.join(td, f"{basename or 'input'}.pptx")
        out_dir = td
        with open(in_path, 'wb') as f:
            f.write(pptx_bytes)
        # Run LibreOffice headless conversion
        cmd = [soffice, '--headless', '--convert-to', 'pdf', '--outdir', out_dir, in_path]
        logging.info(f"Converting PPTX to PDF using LibreOffice: {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            logging.error(proc.stderr.decode(errors='ignore'))
            raise RuntimeError('Failed to convert PPTX to PDF via LibreOffice.')
        # Find output PDF
        expected_pdf = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(in_path))[0]}.pdf")
        if not os.path.exists(expected_pdf):
            # Sometimes LibreOffice may rename; fallback to first pdf in out_dir
            pdf_candidates = [p for p in os.listdir(out_dir) if p.lower().endswith('.pdf')]
            if not pdf_candidates:
                raise RuntimeError('PPTX conversion succeeded but no PDF was produced.')
            expected_pdf = os.path.join(out_dir, pdf_candidates[0])
        with open(expected_pdf, 'rb') as f:
            return f.read()


def _maybe_convert_pptx_kwargs_to_pdf(kwargs: dict[str, Any]) -> dict[str, Any]:
    headers = kwargs.get('headers', {})
    is_pptx, stem = _is_pptx(headers)
    if not is_pptx:
        return kwargs
    pdf_bytes = _convert_pptx_bytes_to_pdf(kwargs['content'], basename=stem or None)
    # Build new headers for PDF
    new_headers = dict(headers)
    new_headers['Content-Type'] = 'application/pdf'
    filename = f"{stem or hashlib.sha256(pdf_bytes).hexdigest()}.pdf"
    new_headers['Content-Disposition'] = f'inline; filename="{filename}"'
    logging.info(
        f"Converted PPTX to PDF ({len(kwargs['content'])} -> {len(pdf_bytes)} bytes)."
    )
    return dict(content=pdf_bytes, headers=new_headers)


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


def docsend_doc_id(url: str) -> str:
    """Extract the DocSend doc path from a view URL.

    Keeps the full path after /view/ so both the simple form
    (/view/<id>) and the spaces form (/view/<space>/d/<doc>) resolve to the
    real deck URL. Taking only the last path segment 404s on spaces URLs.
    """
    if '/view/' in url:
        return url.split('/view/', 1)[-1].split('?')[0].strip('/')
    return url.split('/')[-1]


def generate_pdf_from_docsend_url(url: str,
                                  email: str,
                                  passcode: str | None = None) -> dict[str, Any]:
    inputhash = hash((url, email, passcode))
    cache_key = f"docsend2pdf:{inputhash}"
    if cache.exists(cache_key):
        logging.info(f"Using cache for {url}...")
        return pickle.loads(base64.b64decode(str(cache.get(cache_key))))

    doc_id = docsend_doc_id(url)
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


# Bound OCR cost so a single deck fits the caller's timeout and doesn't
# exhaust memory: lower rasterization DPI and cap the number of pages.
OCR_DPI = int(os.getenv("OCR_DPI", "150"))
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "25"))


def rasterize_pdf_to_images(pdf_path: str, output_folder: str | None = None):
    logging.info(f"Rasterizing PDF {pdf_path} to images (dpi={OCR_DPI}, max_pages={OCR_MAX_PAGES})...")
    images = pdf2image.convert_from_path(  # type: ignore[attr-defined]
        pdf_path, output_folder=output_folder, dpi=OCR_DPI, first_page=1, last_page=OCR_MAX_PAGES)
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
            f"OCR complete, caching {len(kwargs['content'])} bytes under {cache_key}."
        )
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
        temp.flush()
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


def pptx2pdf_sync(req: PptxRequest):
    # Use cache if available
    cache_key = f"pptx2pdf:{req.url}"
    if cache.exists(cache_key):
        logging.info(f"Using cache for {req.url}...")
        kwargs_cached = pickle.loads(base64.b64decode(str(cache.get(cache_key))))
        return make_referenced_response(req.reference, kwargs_cached)

    # Download the PPTX
    file_kwargs = get_or_download_file(req.url)
    headers = cast(dict[str, Any], file_kwargs.get('headers', {}))
    is_pptx, stem = _is_pptx(headers)

    if not is_pptx:
        # Try from URL path
        parsed = urlparse(req.url)
        path = parsed.path or ''
        if path.lower().endswith('.pptx'):
            stem = os.path.splitext(os.path.basename(path))[0]
            is_pptx = True

    if not is_pptx:
        raise ValueError('Provided URL does not appear to be a PPTX file.')

    pdf_bytes = _convert_pptx_bytes_to_pdf(file_kwargs['content'], basename=stem or None)

    filename = f"{(stem or 'presentation')}.pdf"
    headers_out = {
        'Content-Type': 'application/pdf',
        'Content-Disposition': f'inline; filename="{filename}"'
    }

    kwargs = dict(content=pdf_bytes, headers=headers_out)
    logging.info(
        f"Converted {req.url} to PDF, caching {len(kwargs['content'])} bytes under {cache_key}."
    )
    cache.set(cache_key, base64.b64encode(pickle.dumps(kwargs)))
    return make_referenced_response(req.reference, kwargs)
