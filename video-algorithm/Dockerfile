FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3
ENV PYTHONUNBUFFERED 1

RUN chmod 777 /tmp

RUN apt-get update
RUN apt-get -y install python3-tk

ENV LC_ALL=zh_CN.utf8
ENV LANG=zh_CN.utf8
ENV LANGUAGE=zh_CN.utf8
RUN localedef -c -f UTF-8 -i zh_CN zh_CN.utf8

COPY requirements.txt /
RUN pip3 install -r /requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY helmet_detection /ai-alg/helmet_detection_v3
COPY ai_common /ai-alg/helmet_detection_v3/ai_common

WORKDIR /ai-alg/helmet_detection_v3/

CMD ["python3", "helmet_detection_v3.py"]
