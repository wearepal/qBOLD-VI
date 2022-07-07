FROM python:3.8

WORKDIR /qBOLD/qBOLD-VI
ADD . /qBOLD/qBOLD-VI

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "qbold_main.py", "configurations/optimal.yaml"]