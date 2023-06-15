TAG:=latest

build:
	docker build --target=prod -t azoff/flask-pdf-tokenizer:$(TAG) .

publish: build
	docker push azoff/flask-pdf-tokenizer:$(TAG)

server:
	docker compose up server