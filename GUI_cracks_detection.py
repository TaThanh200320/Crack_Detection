import cv2 as cv
import numpy as np
import imutils
import math
from matplotlib import pyplot as plt
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtGui, uic, QtWidgets

def path_img(fileName):
	try:
		img = cv.imread(fileName)
		image = cv.resize(img,None,fx = 0.15, fy = 0.15,interpolation= cv.INTER_AREA)
		global image_orin
		image_orin = image.copy()
		return image
	except:
		pass

def crop_img(image):
	gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	ret,thresh = cv.threshold(gray_img,86,255,cv.THRESH_BINARY)
	kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
	dilate = cv.dilate(thresh,kernel,iterations = 1)
	cnts = cv.findContours(dilate.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = sorted(cnts, key = cv.contourArea, reverse = True)[0]

	rect = cv.minAreaRect(c)
	box = cv.BoxPoints(rect) if imutils.is_cv2() else cv.boxPoints(rect)
	box = np.int0(box)
	draw = cv.drawContours(image, [box], -1, (0, 255, 0), 3)
	transpose = box.transpose()
	xmin = min(transpose[0])
	xmax = max(transpose[0])
	ymin = min(transpose[1])
	ymax = max(transpose[1])
	ymin_offset = 6
	xmin_offset = 260
	xmax_offset = 40
	crop = image_orin[ymin+ymin_offset:ymax-ymin_offset, xmin+xmin_offset:xmax-xmax_offset]
	#cv.imshow('crop',crop)
	return crop, xmin, xmax, ymin, ymax, ymin_offset, xmin_offset, xmax_offset

def delete_circle(crop, xmin, xmax, ymin, ymax):
	rect_crop = crop.copy()
	rect_crop = cv.cvtColor(rect_crop,cv.COLOR_BGR2GRAY)
	template = cv.imread(r'D:\Image_processing_problems\Crack_detection\template\template2.jpg')
	template = cv.resize(template,None,fx = 0.15, fy = 0.15,interpolation= cv.INTER_AREA)
	template = cv.cvtColor(template,cv.COLOR_BGR2GRAY)
	w, h = template.shape[::-1]
	method = cv.TM_CCOEFF_NORMED
	res = cv.matchTemplate(rect_crop,template,method)
	min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
	if method in ['cv.TM_CCOEFF_NORMED']:
		top_left = min_loc
	else:
		top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)
	a_tm = (top_left[0] + bottom_right[0])//2
	b_tm = (top_left[1] + bottom_right[1])//2
	cv.rectangle(rect_crop,top_left, bottom_right, 255, 2)
	fill_crop = crop.copy()
	cv.rectangle(fill_crop, (top_left[0],top_left[1]), (bottom_right[0],bottom_right[1]), (127,127,127), -1)

	#---------------------------------------------------------Morphological_fill_crop--------------------------------------------------------------
	gray_img = cv.cvtColor(fill_crop, cv.COLOR_BGR2GRAY)
	ret,thresh = cv.threshold(gray_img,65,255,cv.THRESH_BINARY)
	convert = cv.bitwise_not(thresh)
	kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
	dilate_fill_crop = cv.dilate(convert,kernel,iterations = 1)
	return dilate_fill_crop, a_tm, b_tm, w, h

def detect(crop, dilate_fill_crop, a_tm, b_tm, w, h, xmin, xmax, ymin, ymax, ymin_offset, xmin_offset, xmax_offset):
	crop_finall = crop.copy()
	crop_fn = crop_finall.copy()
	binary = dilate_fill_crop.copy()
	binary = cv.GaussianBlur(binary, (17, 17), 0)
	contours,hierachy=cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
	height, width = crop_fn.shape[:2]
	for c in range(len(contours)):
		x, y, w, h = cv.boundingRect(contours[c])
		a = x+w//2
		b = y+h//2
		area = cv.contourArea(contours[c])
		if area < 1300:
			continue
		center = (a,b)
		radius = 2
		if math.dist([a, b], [a_tm, b_tm]) < 120:
			cv.drawContours(crop_fn, contours, c, (0, 255, 0), 1, 8)
			cv.rectangle(crop_fn, (x, y), (x+w, y+h), (0, 0, 255), 1, 8, 0)
			cv.putText(crop_fn, 'crack', (x+w+5, y+h-5), cv.FONT_HERSHEY_SIMPLEX,
				fontScale=0.7, color=(0, 175, 255), thickness=2)

	#----------------------------------------------------------Return_original_image-----------------------------------------------------------------
	img1 = image_orin
	img2 = crop_fn
	roi = img1[ymin+ymin_offset:ymax-ymin_offset, xmin+xmin_offset:xmax-xmax_offset]
	img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
	ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
	mask_inv = cv.bitwise_not(mask)
	img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
	img2_fg = cv.bitwise_and(img2,img2,mask = mask)
	dst = cv.add(img1_bg,img2_fg)
	img1[ymin+ymin_offset:ymax-ymin_offset, xmin+xmin_offset:xmax-xmax_offset] = dst
	img1 = cv.resize(img1, (640,480))
	save_path = r'D:\Image_processing_problems\Crack_detection\Crack_detect_tm\result\crack'
	cv.imwrite(save_path + '.png', img1)

def main():	
	class App(QtWidgets.QMainWindow):
		def __init__(self):
			super(App, self).__init__()
			uic.loadUi(r'D:\Image_processing_problems\Crack_detection\Crack_detect_tm\Gui_crack_detect\mygui.ui', self)
			self.button.clicked.connect(self.openFileNameDialog)
			self.setWindowTitle('Crack detection')
		
		def openFileNameDialog(self):
			options = QFileDialog.Options()
			options |= QFileDialog.DontUseNativeDialog
			fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
			path = path_img(fileName)
			cropimg, xmin, xmax, ymin, ymax, ymin_offset, xmin_offset, xmax_offset = crop_img(path)
			deletecircle, a_tm, b_tm, w, h = delete_circle(cropimg, xmin, xmax, ymin, ymax)
			detect(cropimg, deletecircle, a_tm, b_tm, w, h, xmin, xmax, ymin, ymax, ymin_offset, xmin_offset, xmax_offset)
			pixmap = QPixmap(r'D:\Image_processing_problems\Crack_detection\Crack_detect_tm\result\crack.png')
			self.label.setPixmap(pixmap)

	app = QApplication(sys.argv)
	a = App()
	a.show()
	sys.exit(app.exec_())

if __name__ == "__main__":
	main()