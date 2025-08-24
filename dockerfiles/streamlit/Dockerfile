FROM python:3.11-slim

WORKDIR /app

COPY dockerfiles/streamlit/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit/streamlit_app.py ./streamlit_app.py

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]