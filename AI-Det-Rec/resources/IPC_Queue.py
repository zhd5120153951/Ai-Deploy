'''
@FileName   :IPC_Queue.py
@Description:
@Date       :2024/01/05 21:11:02
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import multiprocessing


def test(q):
    for i in range(10):
        q.put(i)   # 将元素放入queue
    q.close()


def get_q(q):
    while not q.empty():    # 判断队列是否为非空，非空才继续执行下一步
        res = q.get()       # 获取一个元素
        print(res)
        # print(q.qsize())


if __name__ == '__main__':
    q = multiprocessing.Queue()      # 定义queue,这里可以传入参数，即队列所含最大元素量
    p1 = multiprocessing.Process(target=test, args=(q,))   # 多进程
    p2 = multiprocessing.Process(target=get_q, args=(q,))
    # 此例仅为介绍用，实际应用时创建的进程可以有丰富的功能，只需要将queue作为一个参数传入进程即可。
    p1.start()
    p2.start()
    p1.join()
    p2.join()
