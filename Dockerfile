FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir \
    tensorflow==2.16.1 keras==3.10.0 fastapi==0.111.0 uvicorn==0.30.1 \
    pillow==10.4.0 numpy==1.26.4 python-multipart==0.0.9
COPY src/ src/
COPY models/model.h5 models/model.h5
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
