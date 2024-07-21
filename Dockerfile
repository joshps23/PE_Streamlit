FROM python:3.11.5 AS stage-one
EXPOSE 8501
WORKDIR /app



RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .

FROM python:3.11.5
WORKDIR /app
COPY --from=stage-one /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=stage-one /app /app

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# CMD streamlit run streamlit_app.py