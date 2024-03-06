
import logging
import multiprocessing_logging
import multiprocessing
import os

# 创建日志器
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 创建一个文件处理器
file_handler = logging.FileHandler('test_log.txt')
file_handler.setLevel(logging.DEBUG)

# 设置日志输出格式
formatter = logging.Formatter(
    '%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将文件处理器添加到根日志器
logger.addHandler(file_handler)


def worker():
    logger.debug(f'Running in process {os.getpid()}')


if __name__ == '__main__':
    # 启用多进程日志记录
    multiprocessing_logging.install_mp_handler()

    # 创建多个进程
    processes = []
    for _ in range(6):
        p = multiprocessing.Process(target=worker)
        processes.append(p)
        p.start()

    # 等待所有进程结束
    for p in processes:
        p.join()
