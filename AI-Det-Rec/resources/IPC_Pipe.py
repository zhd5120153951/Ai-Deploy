'''
@FileName   :IPC_Pipe.py
@Description:
@Date       :2024/01/05 21:10:32
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import multiprocessing


def test1(conn):
    conn.send('test1传送到：jack')      # 传递一个信息‘jack’
    conn.send('test1传送到：mary')	   # 传递另个信息‘mary’
    print(conn.recv())     # 获取从管道的另一端发送来的第一个值
    print(conn.recv())     # 获取从管道的另一端发送来的第二个值


def test2(conn):
    conn.send('test2传送：你好')      # 传递一个信息‘你好’
    conn.send('test2传送：HELLO')    # 传递另个信息‘HELLO’
    print(conn.recv())    # 获取从管道的另一端发送来的第一个值
    print(conn.recv())    # 获取从管道的另一端发送来的第二个值


if __name__ == '__main__':
    p_conn, c_conn = multiprocessing.Pipe()  # 定义一个管道的两端，之后将这两端传出去
    p1 = multiprocessing.Process(target=test1, args=(p_conn,))    # 把一端给test1
    p2 = multiprocessing.Process(target=test2, args=(c_conn,))    # 另一端给test2
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    # 这样就将test1与test2函数进行了管道传输，test1的内容传给了test2，test2的内容传给了test1。
    # test1与test2被分配到了不同的进程，因此这样相当于就进行了多进程中进程间的通信。
