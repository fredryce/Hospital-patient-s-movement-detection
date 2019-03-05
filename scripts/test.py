import cv2
import numpy as np
#import prediction
from multiprocessing import Queue, Process
import os, sys

def run_detect(rec, send):
	cap = cv2.VideoCapture('./../test.mp4')
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	wait_time = int((1/int(fps)) * 1000)
	frame_num = 0
	while True:
		_, frame = cap.read()
		rec.put(frame)
		cv2.imshow('test', frame)

	


def main():
	x = np.array([True, True, False])
	print(np.all(x))
main()