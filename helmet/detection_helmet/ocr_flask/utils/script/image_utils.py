# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 11:18:11 2019

@author: shenfangyuan

pip install pydicom
pip install dicom

1,本程序使用SimpleITK软件处理dicom格式的医学图片
2,dicom格式的医学图片数据类型是int32,所以为了运算,需要转换为float类型
3,数据有"负数值",而且数据动态范围也比较大,这里我用了简单的方法,把数据范围映射到0-1之间
  的浮点数.因为我对数据情况不熟,因此采用了简单的转换算法
  img_tmp = (img_tmp-img_tmp.min())/(img_tmp.max()-img_tmp.min())
  感觉这样处理的图片有点"发白"
4,待数据处理之后,使用img_tmp = (img_tmp*255).astype(np.uint8)转换为opencv可以显示和
  存储的格式


"""
import SimpleITK as sitk
import pydicom
import os
import numpy as np
# import dicom
import matplotlib.pyplot as plt
import cv2

base_dir = '/home/geekplusa/ai/datasets/cv/simpleedu/医院/data/images_dcm'
filename = os.path.join(base_dir, '60F763DC510A59.DCM')


def loadFile(filename):  # 读取dcm文件中的图片内容
    ds = sitk.ReadImage(filename)  # 用SimpleITK读取 dcm格式的文件
    img_array = sitk.GetArrayFromImage(ds)  # 获取文件中图片的raw原始数据
    frame_num, width, height = img_array.shape  # 获取图片的尺寸
    return img_array, frame_num, width, height  # 返回图片及尺寸信息


def loadFileInformation(filename):  # 读取dcm文件中的信息字段说明信息
    information = {}
    ds = pydicom.read_file(filename, force=True)
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['InstitutionName'] = ds.InstitutionName
    information['Manufacturer'] = ds.Manufacturer
    #    information['NumberOfFrames'] = ds.NumberOfFrames
    return information


def dcm_2_jpg(input_filename, output_filename):
    info = loadFileInformation(input_filename)  # 调用函数,获取图片的文件头说明信息,我这里没用
    img_tmp = np.zeros([512, 512])  # 初始化缓冲区域
    img_array, frame_num, width, height = loadFile(input_filename)
    print('frame_num', frame_num, 'width', width, 'height', height)
    # 获取图片内容及维度信息
    img_tmp = img_array[0, :, :]  # 把数据从(1,512,512)转换为(512,512)
    img_tmp = img_tmp.astype(np.float32)  # 把数据从 int32转为 float32类型
    img_tmp = (img_tmp - img_tmp.min()) / (img_tmp.max() - img_tmp.min())
    # 把数据范围变为0--1浮点,或许还有其他转换方法,效果能更好一些.

    img_tmp = (img_tmp * 255).astype(np.uint8)  # 转换为0--256的灰度uint8类型

    # plt.imshow(img_tmp,cmap='gray')
    cv2.imwrite(output_filename, img_tmp)
    # cv2.imshow('img_mha', img_tmp)  #图片显示
    # cv2.waitKey(0)  #等待用户输入
    # cv2.destroyAllWindows()  #注销显示的窗口


def main():
    base_dir = '/home/geekplusa/ai/datasets/cv/simpleedu/医院/data/images_dcm'
    listdir = os.listdir(os.path.join(base_dir, 'dcm'))
    for dcm in listdir:
        input_filename = os.path.join(os.path.join(base_dir, 'dcm'), dcm)
        output_filename = os.path.join(os.path.join(base_dir, 'jpg'), dcm[:-4] + '.jpg')
        dcm_2_jpg(input_filename, output_filename)


if __name__ == '__main__':
    main()
