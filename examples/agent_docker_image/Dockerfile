FROM python:3.10

WORKDIR /src

COPY requirements.txt /src
RUN pip install -Ur requirements.txt

COPY . /src

WORKDIR /src/src

CMD ["serve", "run", "-h", "0.0.0.0", "serve:agent"]
