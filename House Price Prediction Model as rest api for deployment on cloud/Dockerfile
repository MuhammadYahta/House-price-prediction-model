FROM python:3.5-slim
WORKDIR /cc
ADD . /cc
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 5050
ENV NAME OpentoAll
CMD ["python", "server.py"]

