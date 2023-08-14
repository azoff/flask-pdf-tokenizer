FROM python:3-slim-buster as local

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		wkhtmltopdf \
	&& rm -rf /var/lib/apt/lists/*

ADD requirements.txt .
RUN pip3 install --upgrade pip \
	&& pip3 install -vr requirements.txt

ENV WKHTMLTOPDF_PATH=/usr/bin/wkhtmltopdf

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM local as prod

ADD main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]