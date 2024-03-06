'''
@FileName   :IPC_Array.py
@Description:跨进程通信--Array
@Date       :2024/01/03 11:20:05
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

from multiprocessing import Process, Array, Lock


def child_process(shared_arr, lock):
    with lock:
        # 加锁，子进程读取共享数组的值
        print("子进程读取共享数组的值:", list(shared_arr))


def main_process(shared_arr, lock):
    with lock:
        # 加锁，主进程改变共享数组的值
        for i in range(len(shared_arr)):
            # for i in shared_arr:
            shared_arr[i] = True
        print("主进程改变共享数组的值为:", list(shared_arr))


if __name__ == '__main__':
    # 创建共享数组和锁
    # shared_arr = Array('i', range(5))
    shared_arr = Array('b', [False, False])
    # shared_arr
    print("初始化:", list(shared_arr))
    # print("初始化:", shared_arr)

    lock = Lock()

    # 创建子进程
    p = Process(target=child_process, args=(shared_arr, lock))
    p.start()

    # 主进程改变数组的值
    main_process(shared_arr, lock)

    # 等待子进程结束
    p.join()
