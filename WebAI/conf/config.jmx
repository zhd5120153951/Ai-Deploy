from concurrent import futures
import time
import subprocess
import codecs
import sys
import os
import logging
import json
import grpc
import test_pb2
import test_pb2_grpc

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='rpc.log',
                    filemode='a+')

rpc_server = r'192.168.22.137'
rpc_port = '50051'


class performance():

    '''性能测试的客户端接口'''

    def __init__(self, ip, port):
        '''初始化，连接RPC服务'''

        logging.info("performance_client init")
        conn = grpc.insecure_channel(ip + ':' + port)
        self.stub_client = test_pb2_grpc.RPCStub(channel=conn)

    def sendConfig(self, filename):
        file_handle = codecs.open(filename, 'r', encoding='utf-8')
        content = file_handle.read()
        '''向RPC server发送测试的配置文件'''
        response = self.stub_client.sendConfFile(
            test_pb2.Content(text=content))
        print(response.code)


if __name__ == '__main__':
    client = performance(rpc_server, rpc_port)
    # 将本地的rpc_client.py文件当做测试文件发送到server端
    client.sendConfig('test_client.py')
