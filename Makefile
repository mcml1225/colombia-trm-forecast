.PHONY: help install run test clean docker-build docker-run

help:
@echo "Available commands:"
@echo "  make install      - Install dependencies"
@echo "  make run          - Run Streamlit dashboard"
@echo "  make test         - Run tests"
@echo "  make clean        - Clean temporary files"
@echo "  make docker-build - Build Docker image"
@echo "  make docker-run   - Run Docker container"
@echo "  make update-data  - Fetch latest TRM data"
@echo "  make train        - Train all models"

install:
pip install -r requirements.txt
pre-commit install

run:
streamlit run dashboard/app.py

test:
pytest tests/ -v --cov=src

clean:
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pkl" -delete
rm -rf .pytest_cache
rm -rf .coverage
rm -rf htmlcov
rm -rf .mypy_cache

docker-build:
docker build -t trm-forecast:latest .

docker-run:
docker-compose up -d

update-data:
python src/data_fetcher.py

train:
python src/traditional_model.py
python src/ai_model.py
python src/ensemble_model.py

deploy:
@echo "Deploying to production..."
@echo "Add your deployment commands here"
