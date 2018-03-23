# FROM python:3.5-alpine
FROM frolvlad/alpine-python-machinelearning

RUN mkdir /code/
WORKDIR /code/

ADD . .

RUN pip install -r requirements.txt
RUN pip install --upgrade pip

RUN pip install -U nltk

# Download nltk, but this will be very large
RUN python -m nltk.downloader -q all

EXPOSE 5000

CMD ["python", "-m", "main"]