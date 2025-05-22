.PHONY: setup run api download-model docker docker-run clean

setup:
	pip install -r requirements.txt

run:
	./main.py "$(QUERY)"

api:
	uvicorn app:app --reload

download-model:
	./download_model.py

docker:
	docker build -t deepsearch .

docker-run:
	docker-compose up -d

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete