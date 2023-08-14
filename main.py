import tempfile
import logging
import re
import pydantic
import tiktoken
import redis
import os
from fastapi import FastAPI
from urllib.request import urlretrieve
from pdfminer.high_level import extract_text

class Request(pydantic.BaseModel):
	url: str
	extra_context: str = ''

class TruncateRequest(Request):
	max_tokens: int = 2048
	model: str = "gpt-4"

logging.basicConfig(level=logging.INFO)

app = FastAPI()
cache = redis.Redis(host=os.getenv('REDIS_HOST'), port=6379, decode_responses=True)

@app.get("/")
def index():
	return {"status": "ok"}

@app.post("/text")
def text(req:Request):
	text = download_pdf_and_extract_text(req.url, extra_context=req.extra_context)
	return { "text": text }

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
	
	global cache

	if not url:
		logging.warning("No URL provided, returning only the input text.")
		return extra_context

	# hash the inputs into a cache key
	cache_key = f"pdf:{hash((url, extra_context))}:text"
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
