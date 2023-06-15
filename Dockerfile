FROM python:3-slim-buster

WORKDIR /app

ADD requirements.txt .
RUN pip3 install --upgrade pip \
	&& pip3 install -vr requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]