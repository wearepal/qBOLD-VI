FROM python:3.8

WORKDIR /qBOLD/qBOLD-VI
ADD . /qBOLD/qBOLD-VI

RUN pip install -r requirements.txt
#RUN pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk2/centos-7 wxpython
#RUN pip install fsleyes

CMD ["python", "qbold_main.py", "configurations/optimal.yaml"]