"""
For detecting humans, beds, and human egression from said beds.
"""
import os, cv2
import numpy as np
import pandas as pd
import prediction
from skimage.measure import compare_ssim as ssim
from tf_pose.common import CocoPart

# combine the use of cnn and rnn. the cnn determines the bounding box, based on the iou value of the binding box and the bed over time. 
# we feed iou value over time into the rnn(lstm) to find if the iou is determined to be trying to get out of bed or not.

#print(os.getcwd()) #//actual working directory is the root, not the scripts directory

class VDT(object):

	def __init__(self, write, convert_csv, source=0):
		self.write = write
		self.convert_csv = convert_csv

		self.out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
		cv2.namedWindow('window', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('window', (960, 720))
		self.detection(cv2.VideoCapture(source))

		#print('video saved to ', os.path.abspath(videoOut))


	def debug(self, prev_frame, frame):
		try:
			if prev_frame.any():
				"""
				prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
				frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				diff_score = ssim(prev_gray, frame_gray)
				print('Image difference Score:', diff_score)
				"""
		except AttributeError as e:
			print(e)


	def detection(self, cap):

		detection = prediction.OpenPose('resources/models/cmu/cmu.pb', convert_csv=self.convert_csv)
		prev_frame = None

		while cap.isOpened() and cv2.getWindowProperty('window', 0) >= 0:

			ret, frame = cap.read()
			if not ret: break
			#self.debug(prev_frame, frame)

			result = detection.detect(frame)

			frame = detection.draw_humans(frame, result, imgcopy=False)
			cv2.imshow('window', frame)
			
			if self.write: self.out.write(frame)
			prev_frame = frame
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
		
		cap.release()
		self.out.release()
		cv2.destroyAllWindows()

		if self.convert_csv:
			df_final = pd.concat([detection.df, pd.DataFrame(detection.temp_list, columns=detection.df.columns.values)], ignore_index=True)
			df_final.to_csv('resources/timeseries.csv')



if __name__ == '__main__':
	VDT(write=True, convert_csv=False, source='getOut.mp4')
