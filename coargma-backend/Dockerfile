FROM python:3.10.12
COPY . /app_backend
WORKDIR /app_backend
RUN apt-get upgrade && apt-get install make
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN make /app_backend
EXPOSE 15555
CMD [ "python", "demo_api_flask.py" ]
