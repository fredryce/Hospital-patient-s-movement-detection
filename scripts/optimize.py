import cv2
import numpy as np
import prediction
from multiprocessing import Queue, Process
import os, sys

#problem the detection frame is still too far behind, and when its taking priority, its offsetting the ponits too much. how to calculate which one is more accurate one

os.chdir("./../")
ROOT_DIR = os.path.abspath(os.getcwd())


def run_detection(recieve_que, send_que, after_frame_que):
    bed = ((290, 230),(720,230),(100, 720),(950, 720))
    lk_params = dict(winSize = (15, 15),
                     maxLevel = 6,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
     
    detection = prediction.OpenPose(os.path.join(ROOT_DIR, 'visual-detection/resources/models/cmu/cmu.pb'), convert_csv=False,bed=bed)
    while True:
        if not recieve_que.empty():
            print('im gettting frame')
            frame, frame_number = recieve_que.get()
            result = detection.detect(frame)
            output, center = detection.draw_humans(frame, result, imgcopy=True)
            cv2.imwrite('testout.jpg', output)
            try:
                if center.any():
                    send_que.put((center, frame_number, output))
            except AttributeError as e:
                send_que.put((np.array([[],[],[]]), frame_number))
            print('done processing')


def main():
    cap = cv2.VideoCapture('test.mp4')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    wait_time = int((1/int(fps)) * 1000)
    # Create old frame
    _, frame = cap.read()
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
    # Lucas kanade params
    lk_params = dict(winSize = (25, 25),
                     maxLevel = 4,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    point_selected = False
    stop_detection = False
    old_points = np.array([[]])
    frame_num = 1
    send_que, recieve_que, after_frame_que = Queue(), Queue(), Queue()
    recieve_que.put((frame,0))
    p = Process(target=run_detection, args=(recieve_que, send_que, after_frame_que), daemon=True)
    p.start()

    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.waitKey(wait_time)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if point_selected is False and stop_detection is False: #initial running the detection points is not picked and stop detection false
            recieve_que.put((frame, frame_num))
            stop_detection = True
            stopped_frame = frame_num + 1



            
        if not send_que.empty(): 
            center, frame_number, frame= send_que.get()
            print(center)
            tracked_points = np.copy(center)
            tracked_points[tracked_points!=np.array([None, None])]
            print(tracked_points)
            point_selected = True
            old_points = center

            
        if point_selected:
            if frame_num % 3 ==0:
                point_selected = False
                stop_detection = False
            new_points = center

        
        '''

        else:
            try:
                new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
            except Exception as e:
                print(e)

        try:
            old_gray = gray_frame.copy()
            old_points = new_points
            x1, y1,x2,y2,x3,y3 = new_points.ravel()

            cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x3, y3), 5, (0, 255, 0), -1)
        except Exception as e:
            pass
            
        '''
        cv2.imshow("Frame", frame)

        frame_num += 1

     
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()