# -*- coding: utf-8 -*-
import shutil
from xml.dom.minidom import Document
import os
import cv2


# def makexml(txtPath, xmlPath, picPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
def makexml(picPath, txtPath, xmlPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
    """此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
    """
    dic = {'0': "0",  # 创建字典用来对类型进行转换
           '1': "1",  # 此处的字典要与自己的classes.txt文件中的类对应，且顺序要一致
           '2' :'2',
           '3' :'3',
           '4' :'4',
           '5' :'5',
           '6' :'6',
           '7' :'7',
           '8' :'8',
           }
    dict_temp = {}
    files = os.listdir(txtPath)
    for i, name in enumerate(files):
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        txtFile = open(txtPath + name)
        txtList = txtFile.readlines()
        image_filename = picPath + name[0:-4] + ".png"
        if os.path.exists(image_filename):
            try:
                img = cv2.imread(image_filename)
                Pheight, Pwidth, Pdepth = img.shape
            except Exception as e:
                print(image_filename)
                continue
            # 复制到新文件夹
            new_images_path = os.path.join(base_dir, 'process/images/')
            new_labels_path = os.path.join(base_dir, 'process/labels/')
            
            if not os.path.exists(new_images_path):
                print(new_images_path)
                os.makedirs(new_images_path)
            if not os.path.exists(new_labels_path):
                print(new_labels_path)
                os.makedirs(new_labels_path)
            
            shutil.copyfile(image_filename, new_images_path + name[0:-4] + ".png")
            shutil.copyfile(txtPath + name, new_labels_path + name)

            folder = xmlBuilder.createElement("folder")  # folder标签
            foldercontent = xmlBuilder.createTextNode("driving_annotation_dataset")
            folder.appendChild(foldercontent)
            annotation.appendChild(folder)  # folder标签结束

            filename = xmlBuilder.createElement("filename")  # filename标签
            filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
            filename.appendChild(filenamecontent)
            annotation.appendChild(filename)  # filename标签结束

            size = xmlBuilder.createElement("size")  # size标签
            width = xmlBuilder.createElement("width")  # size子标签width
            widthcontent = xmlBuilder.createTextNode(str(Pwidth))
            width.appendChild(widthcontent)
            size.appendChild(width)  # size子标签width结束

            height = xmlBuilder.createElement("height")  # size子标签height
            heightcontent = xmlBuilder.createTextNode(str(Pheight))
            height.appendChild(heightcontent)
            size.appendChild(height)  # size子标签height结束

            depth = xmlBuilder.createElement("depth")  # size子标签depth
            depthcontent = xmlBuilder.createTextNode(str(Pdepth))
            depth.appendChild(depthcontent)
            size.appendChild(depth)  # size子标签depth结束

            annotation.appendChild(size)  # size标签结束

            for j in txtList:
                oneline = j.strip().split(" ")
                object = xmlBuilder.createElement("object")  # object 标签
                picname = xmlBuilder.createElement("name")  # name标签
                dict_temp[oneline[0]] = 'a'
                namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
                picname.appendChild(namecontent)
                object.appendChild(picname)  # name标签结束

                pose = xmlBuilder.createElement("pose")  # pose标签
                posecontent = xmlBuilder.createTextNode("Unspecified")
                pose.appendChild(posecontent)
                object.appendChild(pose)  # pose标签结束

                truncated = xmlBuilder.createElement("truncated")  # truncated标签
                truncatedContent = xmlBuilder.createTextNode("0")
                truncated.appendChild(truncatedContent)
                object.appendChild(truncated)  # truncated标签结束

                difficult = xmlBuilder.createElement("difficult")  # difficult标签
                difficultcontent = xmlBuilder.createTextNode("0")
                difficult.appendChild(difficultcontent)
                object.appendChild(difficult)  # difficult标签结束

                bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
                xmin = xmlBuilder.createElement("xmin")  # xmin标签
                mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
                xminContent = xmlBuilder.createTextNode(str(mathData))
                xmin.appendChild(xminContent)
                bndbox.appendChild(xmin)  # xmin标签结束

                ymin = xmlBuilder.createElement("ymin")  # ymin标签
                mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
                yminContent = xmlBuilder.createTextNode(str(mathData))
                ymin.appendChild(yminContent)
                bndbox.appendChild(ymin)  # ymin标签结束

                xmax = xmlBuilder.createElement("xmax")  # xmax标签
                mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
                xmaxContent = xmlBuilder.createTextNode(str(mathData))
                xmax.appendChild(xmaxContent)
                bndbox.appendChild(xmax)  # xmax标签结束

                ymax = xmlBuilder.createElement("ymax")  # ymax标签
                mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
                ymaxContent = xmlBuilder.createTextNode(str(mathData))
                ymax.appendChild(ymaxContent)
                bndbox.appendChild(ymax)  # ymax标签结束

                object.appendChild(bndbox)  # bndbox标签结束

                annotation.appendChild(object)  # object标签结束
            if not os.path.exists(xmlPath):
                print(xmlPath)
                os.makedirs(xmlPath)
            f = open(xmlPath + name[0:-4] + ".xml", 'w')
            xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
            f.close()
    print(dict_temp)


if __name__ == "__main__":
    base_dir = '/home/geekplusa/ai/datasets/cv/simpleedu/医院/data'
    picPath = os.path.join(base_dir, '原始数据/images/')  # 图片所在文件夹路径，后面的/一定要带上
    txtPath = os.path.join(base_dir, '原始数据/labels/')  # txt所在文件夹路径，后面的/一定要带上
    xmlPath = os.path.join(base_dir, 'process/xml/')  # xml文件保存路径，后面的/一定要带上
    makexml(picPath, txtPath, xmlPath)

