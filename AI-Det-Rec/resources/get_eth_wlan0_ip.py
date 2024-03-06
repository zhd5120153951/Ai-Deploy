'''
@FileName   :get_eth_wlan0_ip.py
@Description:
@Date       :2024/01/25 14:25:41
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import socket
import subprocess

# 获取网口IP地址


def get_eth_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("192.168.0.1", 1))
    ip = s.getsockname()[0]
    s.close()
    return ip

# 获取WiFi IP地址


def get_wifi_ip():
    wifi_ip = ""
    # with open("/sys/class/net/wlan0/address") as f:
    #     wifi_ip = f.read().strip()
    wifi_ip = "192.168.22.137"
    return wifi_ip

# 获取IP地址信息


def get_ips():
    eth_ip = get_eth_ip()
    wifi_ip = get_wifi_ip()
    ip_info = {
        "网口IP地址": eth_ip,
        "WiFi IP地址": wifi_ip,
    }
    return ip_info


if __name__ == "__main__":
    ip_info = get_ips()
    for key, value in ip_info.items():
        print(key + ": " + value)
