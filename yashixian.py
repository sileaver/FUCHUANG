import time
from tracker import *

from AIDetector_pytorch import Detector
import imutils

import cv2
fps=0.0
import numpy as np


points_list=[]
image=None
def point_distance_line(point,line_point1,line_point2):
	#计算向量
	vec1 = line_point1 - point
	vec2 = line_point2 - point
	distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
	return distance

#lane detection
def canny(frame):
	gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	blur=cv2.GaussianBlur(gray,(5,5),0)
	canny=cv2.Canny(blur,100,150)
	# cv2.imshow("canny",canny)
	return canny

def region_of_interest(frame):

	polygons=np.array([
					  [(1,628),(867,192),(1008,304),(1245,347),(1906,777)]
					   ])
	mask=np.zeros_like(frame)
	""" cv2.imshow('abc',mask)"""
	cv2.fillPoly(mask,polygons,255)


	masked_image=cv2.bitwise_and(frame,mask)
	return masked_image
def region_of_interest1(frame):

	polygons=np.array([[(1,628),(867,202),(1768,710)]])

	mask=np.zeros_like(frame)

	# cv2.imshow('abc',frame)
	cv2.fillPoly(mask,polygons,(255,255,255))


	masked_image=cv2.bitwise_and(frame,mask)

	return masked_image
def display_lines(frame,lines):
	global x1,y1,x2,y2
	line_image=np.zeros_like(frame)

	if lines is not None:
		for line in lines:
			x1,y1,x2,y2=line.reshape(4)
			cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
	points_list.append([x1, y1, x2, y2])
	return line_image

def judge(image,bboxes, lines,  line_thickness=None):
	for box in bboxes:
		for line in lines:
			if box:
				if len(box) == 6:
					x1, y1, x2, y2, cls_id, pos_id = box
					x = (x1 + x2) / 2
					y = (y1 + y2) / 2
					x3, y3, x4, y4 = line.reshape(4)
					point = np.array([x, y])
					line_point1 = np.array([x3, y3])
					line_point2 = np.array([x4, y4])
					distance = point_distance_line(point, line_point1, line_point2)
					if distance < 30:
						tl = line_thickness or round(
							0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness

						color = (255, 0, 255)
						c1, c2 = (x1, y1), (x2, y2)

						cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
						tf = max(tl - 1, 1)  # font thickness
						t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
						c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
						cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled

						cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
									[225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
				elif len(box) == 8:
					x1, y1, x2, y2, cls_id, pos_id, speed, start = box
					x = (x1 + x2) / 2
					y = (y1 + y2) / 2
					x3, y3, x4, y4 = line.reshape(4)
					point = np.array([x, y])
					line_point1 = np.array([x3, y3])
					line_point2 = np.array([x4, y4])
					distance = point_distance_line(point, line_point1, line_point2)
					if distance < 30:
						tl = line_thickness or round(
							0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness




						color = (255, 0, 255)
						c1, c2 = (x1, y1), (x2, y2)

						cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
						tf = max(tl - 1, 1)  # font thickness
						t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
						c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
						cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled

						cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
									[225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
# point = np.array([5,2])
# line_point1 = np.array([2,2])
# line_point2 = np.array([3,3])
#
# print(point_distance_line(point,line_point1,line_point2))
#create VideoCapture object and read from video file
# cap = cv2.VideoCapture('dataset/cars.mp4')

#use trained cars XML classifiers
#car_cascade = cv2.CascadeClassifier('cars.xml')

#read until video is completed

def main():
	global fps
	name = 'demo'

	det = Detector()
	cap = cv2.VideoCapture(r'C:\Users\a\Yolov5-Deepsort\data\video\gta2moify_fps_rate.mp4')

	videoWriter = None

	while True:

		# try:
		ret, im = cap.read()
		if im is None:
			break


		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		lane_image = np.copy(im)
		canny2 = canny(lane_image)


		cropped_image = region_of_interest(canny2)

		lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 350, np.array([]), minLineLength=200, maxLineGap=300)
# lines=cv2.HoughLines(cropped_image,2,np.pi/180,100)
		"""averaged_lines=average_slope_intercept(lane_image,lines)"""
		line_image1 = display_lines(lane_image, lines)
		line_image=region_of_interest1(line_image1)

		t1=time.time()

		result = det.feedCap(im)
		bboxes = result['before_box']
		judge(im, bboxes, lines)
		result = result['frame']


		cv2.putText(result, 'fps-{}'.format(fps), (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 1,
					[0, 0, 255], thickness=3, lineType=cv2.LINE_AA)
		result=cv2.addWeighted(result,0.8,line_image,1,1)
		# result = imutils.resize(result, height=500)
		fps = (fps + (1. / (time.time() - t1))) / 2  # 此处的time.time()就是检测完这张图片的结束时间,除以2是为了和之前的fps求一个平均
		print("fps= %.2f"%fps)

		if videoWriter is None:
			fourcc = cv2.VideoWriter_fourcc(
				'm', 'p', '4', 'v')  # opencv3.0
			videoWriter = cv2.VideoWriter(
				'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

		videoWriter.write(result)
		cv2.imshow(name, result)
		t = int(1000 / fps)
		cv2.waitKey(t)

		if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
			# 点x退出
			break
		# except Exception as e:
		#     print(e)
		#     break

	cap.release()
	videoWriter.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
