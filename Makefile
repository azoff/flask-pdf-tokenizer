TAG:=latest

build:
	docker build --target=prod -t azoff/flask-pdf-tokenizer:$(TAG) .

publish: build
	docker push azoff/flask-pdf-tokenizer:$(TAG)
ifneq  ($(TAG),latest)
	docker push azoff/flask-pdf-tokenizer:latest
	git tag v$(TAG)
	git push --tags
endif

server:
	docker compose up server