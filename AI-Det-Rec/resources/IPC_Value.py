'''
@FileName   :IPC_Value.py
@Description:跨进程通信--Value
@Date       :2024/01/03 11:20:34
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import ctypes
from multiprocessing import Process, Value, Lock


def child_process(shared_var, lock):
    with lock:
        # 加锁，子进程读取共享变量的值
        print("子进程读取共享变量的值:", shared_var.value)


def main_process(shared_var, lock):
    with lock:
        # 加锁，主进程改变共享变量的值
        shared_var.value = 10
        print("主进程改变共享变量的值为10")


if __name__ == '__main__':
    # 创建共享变量和锁
    # shared_var = Value(ctypes.c_char_p, "daito".encode("utf-8"))
    shared_var = Value(ctypes.c_int, 0)

    print("初始化值为", shared_var.value)
    lock = Lock()

    # 创建子进程
    p = Process(target=child_process, args=(shared_var, lock))
    p.start()

    # 主进程改变变量的值
    # main_process(shared_var, lock)
    with lock:
        # 加锁，主进程改变共享变量的值
        # shared_var.value = "happy new year".encode("utf-8")
        shared_var.value = 10

        print("主进程改变共享变量的值为:", shared_var.value)
    # 等待子进程结束
    p.join()
