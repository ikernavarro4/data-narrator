FROM python:3.11-slim

LABEL maintainer="Iker Navarro <Inavar11@itam.mx>"
LABEL description="datanarrator — convierte DataFrames en análisis en lenguaje natural"
LABEL version="0.1.5"

WORKDIR /app

RUN pip install --upgrade pip --no-cache-dir && \
    pip install datanarrator --no-cache-dir

COPY examples/ ./examples/

CMD ["python", "examples/ejemplo_titanic.py"]
