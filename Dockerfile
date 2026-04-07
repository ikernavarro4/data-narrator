FROM python:3.11-slim

WORKDIR /app

RUN pip install datanarrator

COPY examples/ ./examples/

CMD ["python", "examples/ejemplo_titanic.py"]
