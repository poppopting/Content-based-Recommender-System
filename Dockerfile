FROM python:3.7
COPY . .
RUN curl https://bootstrap.pypa.io/get-pip.py | python && pip install --upgrade setuptools
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python app.py



