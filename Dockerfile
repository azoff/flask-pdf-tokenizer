FROM python:3.10.17-slim-bookworm AS local

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		libjpeg-dev \ 
		libpng-dev \
		poppler-utils \
		tesseract-ocr \
		wkhtmltopdf \
		zlib1g-dev \
	&& rm -rf /var/lib/apt/lists/*

ADD requirements.txt .
RUN pip3 install --upgrade pip \
	&& pip3 install -vr requirements.txt

ENV WKHTMLTOPDF_PATH=/usr/bin/wkhtmltopdf
ENV XDG_RUNTIME_DIR=/tmp

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM local AS prod

ADD *.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]