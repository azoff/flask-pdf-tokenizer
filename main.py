import tempfile
import logging
import re
import pdfkit
import pydantic
import tiktoken
import redis
import hashlib
import os
from fastapi import FastAPI, Response
from urllib.request import urlretrieve
from pdfminer.high_level import extract_text
from typing import Union

class RenderRequest(pydantic.BaseModel):
	html: str

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
	cache.set(f"html:{digest}", html)
	return { "digest" : digest }

@app.get("/pdf/{digest}")
def pdf(digest:str):
	html = cache.get(f"html:{digest}")
	if (html is None):
		return Response(status_code=404)
	headers = {'Content-Disposition': f'inline; filename="{digest}.pdf"', 'Content-Type': 'application/pdf'}
	pdf = pdf_from_html(html)
	resp = Response(content=pdf, headers=headers)
	return resp

@app.post("/truncate")
def truncate(req:TruncateRequest):
	text = download_pdf_and_truncate_text(
		req.url, 
		extra_context=req.extra_context, 
		max_tokens=req.max_tokens,
		model=req.model
	)
	return { "text": text }

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
		if '/pdf/' in url:
			logging.info('detected internal pdf url, using pdfkit...')
			digest = url.split('/pdf/')[1]
			html = cache.get(f"html:{digest}")
			pdf_from_html(html, temp.name)
		else:
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
