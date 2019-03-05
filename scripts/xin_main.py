import cv2
import numpy as np
import os
from skimage.measure import compare_ssim as ssim
import prediction
import pandas as pd
from tf_pose.common import CocoPart
from multiprocessing import Queue, Process
import mutli_step
ROOT_DIR = os.path.abspath(os.chdir('./../'))
#combine the use of cnn and rnn. the cnn determines the bounding box, based on the iou value of the binding box and the bed over time. we feed iou value over time into the rnn(lstm) to find if the iou is determined to be trying to get out of bed or not.


class VideoWriter(object):

	def __init__(self, cap, videoOut = 'out.mp4'):
		self.fps = int(cap.get(cv2.CAP_PROP_FPS))
		self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.out = cv2.VideoWriter(videoOut,self.fourcc, self.fps, (width, height), True)
		print('video saved to ', os.path.abspath(videoOut))


	def running_on_video(self,frame):
		self.out.write(frame)


class VDT(object):
	def __init__(self, bed, algo = 'pose', source=0, convert_csv=False, save_video=False):
		self.cap = cv2.VideoCapture(source)
		if algo.upper() == 'POSE':
			self.detection = prediction.OpenPose('resources/models/cmu/cmu.pb', lstm_model_file='resources/no.h5', convert_csv=convert_csv, bed = bed)
			#test2 sigmoid, test1 regular, test3 linear
		if save_video:
			self.video_write = VideoWriter(cap)
		self.save_video = save_video
		self.convert_csv = convert_csv
		self.outbed = False


	def frame_detection(self):
		cv2.namedWindow('output')
		cv2.setMouseCallback('output',self.mouse_cb)
		old_points = np.array([[]])
		prev_output = 0
		try:
			_, old_frame = self.cap.read()
			old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
		except Exception as e:
			print('No data coming from source..')
			exit()
	
		lstm_input = np.zeros(shape=(3,36))
		frame_number = 0

		while self.cap.isOpened():
			r, frame = self.cap.read()
			if not r:
				print("all Frames completed")
				break
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#print("difference in image is ", self.find_difference_score(old_gray, gray))

			frame, formatted_data = self.get_result_frame(self.detection, frame, 0)


			if formatted_data.any():

				'''
				bool_list = np.any(lstm_input, axis=1)
				if not np.all(bool_list):#if a space in the array still open
					location = np.where(bool_list==False)[0][0]
					lstm_input[location] = formatted_data
					print('adding to ', location)
					if np.all(np.any(lstm_input, axis = 1)): #checking again if its filled then run lstm
						print('predicting')
						result_out = self.detection.predict_next(lstm_input.reshape(1,3,36))
						prev_output = np.argmax(result_out)

				else:
					lstm_input = np.roll(lstm_input, 2, axis=0)
					lstm_input[-1] = formatted_data
					result_out = self.detection.predict_next(lstm_input.reshape(1,3,36))
					prev_output = np.argmax(result_out)
				print(prev_output)

				'''
				
				result_out = self.detection.predict_next(formatted_data)
				prev_output = np.argmax(result_out)
				if prev_output == 1:
					print('im getting out of bed')
					#find the diff in prob and set threshhold, more data, and more time steps
					#print(result_out[0,1] - result_out[0,0])
				else:
					print("im in bed!!")
				
			cv2.putText(frame,str(self.outbed),(0,100), cv2.FONT_HERSHEY_SIMPLEX, 4,(0,0,0),2,cv2.LINE_AA)

			cv2.imshow('output', frame)
			if self.save_video: self.video_write.running_on_video(frame)
			old_gray = gray
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
		self.cap.release()
		cv2.destroyAllWindows()
		if self.convert_csv:
			print('Writing to CSV..')
			df_final = pd.concat([self.detection.df, pd.DataFrame(self.detection.temp_list, columns=self.detection.df.columns.values)], ignore_index=True)
			df_final.to_csv('./resources/timeseries.csv', index=False)

	def get_result_frame(self, detection, frame, prev_output):
		result = detection.detect(frame)
		img, _, formatted_data = detection.draw_humans(frame, result, prev_output=prev_output,imgcopy=True, outbed=self.outbed)
		return img, formatted_data

	def mouse_cb(self,event, x, y, flags, params):
		if event == cv2.EVENT_LBUTTONDOWN:
			print('button clicked')
			if self.outbed:
				self.outbed = False
			else:
				self.outbed = True


	def find_difference_score(self, old_frame, new_frame):
		return (1-ssim(old_frame, new_frame))*100



if __name__ == '__main__':
	bed = ((290, 230),(720,230),(100, 720),(950, 720))
	VDT(bed, source='test.mp4', convert_csv=False).frame_detection()


