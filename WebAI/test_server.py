from concurrent import futures
import time
import subprocess
import codecs
import sys
import os

import grpc
import test_pb2
import test_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
# server端文件保存的位置
jmeter_config = os.path.join(os.getcwd(), r'conf/config.jmx')


class Performance(test_pb2_grpc.RPCServicer):

    def sendConfFile(self, content, context):
        ''' 保存配置文件,如config.jmx '''
        text = content.text
        try:
            print(jmeter_config)
            conf_handle = codecs.open(jmeter_config, 'w', encoding='utf-8')
            conf_handle.write(text)
            return test_pb2.Status(code=0)
        except Exception as e:
            print(e)
            return test_pb2.Status(code=1)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    test_pb2_grpc.add_RPCServicer_to_server(Performance(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
