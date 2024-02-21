# detection_ocr
### 安装yolov5环境
~~~
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install requirements.txt
~~~
### 安装依赖
~~~
pip install -r requirements.txt
~~~
### 修改配置
~~~
proconf.conf 文件中配置修改
~~~
### 安装启动nginx，替换nginx.conf文件
~~~
# 改成自己的目录
root   /opt/projects/front/dist/;
~~~
### 启动
1. 启动ocr服务
~~~
python ocr_flask/main_v2.py
~~~


### 浏览器运行
[http://localhost:5000](http://localhost:5000)