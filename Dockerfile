FROM python:3.9-slim

WORKDIR /app

COPY . /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["gunicorn", "--bind", "0.0.0.0:80", "main:app"]