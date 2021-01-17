FROM python:3.7

RUN apt-get update -y && \
    apt-get install -y python3-opencv

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

COPY . /

CMD ["gunicorn", "-w 4", "-b", "0.0.0.0:80", "--timeout", "3000",  "app:app"]
