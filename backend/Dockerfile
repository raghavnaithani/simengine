# Use a slim Python base to save space
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Install browser requirements for the scraper
RUN playwright install-deps
RUN playwright install
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]