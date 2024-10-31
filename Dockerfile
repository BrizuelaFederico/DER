FROM python:3.12-slim-bullseye
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ./app/src ./src
CMD [ "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
