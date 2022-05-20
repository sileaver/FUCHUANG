import cv2
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

def display_lines(frame,lines):
	global x1,y1,x2,y2
	line_image=np.zeros_like(frame)
	print(lines)
	if lines is not None:
		for line in lines:
			x1,y1,x2,y2=line.reshape(4)
			cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
	points_list.append([x1, y1, x2, y2])
	return line_image


point = np.array([5,2])
line_point1 = np.array([2,2])
line_point2 = np.array([3,3])

print(point_distance_line(point,line_point1,line_point2))
#create VideoCapture object and read from video file
# cap = cv2.VideoCapture('dataset/cars.mp4')
cap = cv2.VideoCapture(r'D:\Vehicle-Detection-And-Speed-Tracking\Car_Opencv\gta2.mp4')
#use trained cars XML classifiers
#car_cascade = cv2.CascadeClassifier('cars.xml')

#read until video is completed
while True:
	#capture frame by frame
	ret, frame = cap.read()
	#convert video into gray scale of each frames
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	lane_image=np.copy(frame)
	canny2=canny(lane_image)


	cropped_image=region_of_interest(canny2)
	cv2.imshow("crop",cropped_image)

	lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,350,np.array([]),minLineLength=200,maxLineGap=300)
	# lines=cv2.HoughLines(cropped_image,2,np.pi/180,100)
	"""averaged_lines=average_slope_intercept(lane_image,lines)"""
	line_image=display_lines(lane_image,lines)
	frame=cv2.addWeighted(lane_image,0.8,line_image,1,1)
	#cv2.imshow('result',combo_image)
	#cv2.waitKey(0)

	#lane detection ends
	#display the resulting frame

	cv2.imshow('video', frame)
	#press Q on keyboard to exit
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break
#release the videocapture object
cap.release()
#close all the frames
cv2.destroyAllWindows()
