import os


def result_path(root_path, file_path):
    # root_path = os.path.abspath(root_path)
    # file_path = os.path.abspath(file_path)
    save_dir = os.path.join(root_path, file_path)
    '''
    transform the windows path to linux path
    原因：使用os.path.join()在windows系统下使用\作为连接符；在linux系统下使用/作为连接符
    正常系统下路径连接符为/
    windows下不正常
    '''
    save_dir = os.path.abspath(save_dir)
    save_dir = save_dir.replace('\\', '/')  # \\转移为\
    # os.makedirs(save_dir, exist_ok=True)
    return save_dir
