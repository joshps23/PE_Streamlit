FROM python:3.11.5 AS stage-one
COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --no-cache-dir -r requirements.txt

FROM python:3.11.5-slim
WORKDIR /app
COPY --from=stage-one /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=stage-one /app /app
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]