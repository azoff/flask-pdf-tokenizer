TAG:=latest

build:
	docker compose build --target=prod -t azoff/flask-pdf-tokenizer:$(TAG) .

publish: build
	docker push azoff/flask-pdf-tokenizer:$(TAG)