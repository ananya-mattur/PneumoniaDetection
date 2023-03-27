FROM python:3.8.10
COPY . /app
WORKDIR /app
RUN pip intall -r reqirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app 