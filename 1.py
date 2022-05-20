# import time
# from hyperlpr import *
# time_struct = time.localtime(time.time())# 从返回浮点数的时间戳方式向时间元组转换，只要将浮点数传递给如localtime之类的函数。
# #输出结果是struct_time
# print(time_struct)
# print(time_struct.tm_year) 	#4位数年
# print(time_struct.tm_mon)     #月 1到12
# print(time_struct.tm_mday)    #日 1到31
# print(time_struct.tm_hour)	#小时 0到23
# print(time_struct.tm_min)	    #分钟 0到59
# print(time_struct.tm_sec)	    #秒 0到61 (60或61 是闰秒)
# print(time_struct.tm_wday)	#一周的第几日 0到6 (0是周一)
# print(time_struct.tm_yday)	#一年的第几日 1到366 (儒略历)
# print(time_struct.tm_isdst)	#夏令时 -1, 0, 1, -1是决定是否为夏令时的旗帜
# time = time_struct.tm_min
#
import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

img=cv2.imread(r'C:\Users\a\Yolov5-Deepsort\tes.png')
cv2.rectangle(img, (140,60), (150,90), (0,0,255), -1, cv2.LINE_AA)
img = cv2ImgAddText(img, "大家好，我是星爷", 140, 60, (255, 255, 0), 20)

cv2.imshow("result",img)
cv2.waitKey(0)