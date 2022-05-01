FROM python:3.8-slim

WORKDIR /app

COPY HW.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["HW.py"]