'''
@FileName   :is_validate_ip.py
@Description:判断一个IP地址是否合规
@Date       :2024/01/25 11:28:18
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import re


def is_valid_ip(ip_address):
    # 使用正则表达式匹配IP地址的格式
    pattern = '^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    if re.match(pattern, ip_address):
        return True
    else:
        return False


# 测试示例
ip1 = "192.168.0.1"
ip2 = "256.0.0.1"
ip3 = "192.168.0"
ip4 = "abc.def.ghi.jkl"
ip5 = "192.168"


print(is_valid_ip(ip1))  # 输出：True
print(is_valid_ip(ip2))  # 输出：False
print(is_valid_ip(ip3))  # 输出：False
print(is_valid_ip(ip4))  # 输出：False
print(is_valid_ip(ip5))  # 输出：False
