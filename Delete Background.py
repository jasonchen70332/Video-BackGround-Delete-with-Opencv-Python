"""
Delete Background
"""

import numpy as np
import cv2
import time

debug = 1
dilate_num = 2
erode_num = 1

#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('C:\\Users\\User8888\\Desktop\\video.mp4')
cap = cv2.VideoCapture('C:\\Users\\User8888\\Desktop\\Python Learn\\Python_code\\cam1.mp4')

# built background space
# MOG2 can rewrite the parameter in the function
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = np.ones((3, 3),np.uint8) #/25

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID') #!!! choose fourcc
# For Gray Level Video with "False"!!!
out = cv2.VideoWriter('Background.avi', fourcc , 30.0, (720,480),False)

# For RGB Video
#out = cv2.VideoWriter('Background.avi', fourcc , 30.0, (720,480),true)

start = time.time()
i = 0
# Read until video is completed
while(True):
    # Capture frame-by-frame
    i = i + 1
    [ret, frame] = cap.read()
    
    if ret == True:
        # Gaussion Filter
        #blur = cv2.blur(frame,(5,5))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
        #fgmask = fgbg.apply(frame)
        fgmask = fgbg.apply(gray)
        thresh = fgmask
        # threshold binary 200~255 --> 255
        thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        
        # dilate, erode thresh
        #thresh = cv2.dilate(thresh, kernel, iterations = dilate_num) 
        thresh = cv2.erode(thresh, kernel, iterations = erode_num)
        thresh = cv2.dilate(thresh, kernel, iterations = dilate_num) 
        thresh = cv2.erode(thresh, kernel, iterations = erode_num)
        """
        # For SURF test
        surf = cv2.xfeatures2d.SURF_create()
        [kp, des] = surf.detectAndCompute(thresh, None)
        frame = cv2.drawKeypoints(thresh, kp, frame)
        """
        '''
        for i in range(0,720):
            for j in range(0,480):
                if fgmask[j,i] < 255:
                    fgmask[j,i] = 0
        '''
        
        # optical flow
        #cv2.calcOpticalFlowFarneback()
        
        # write video
        #out.write(thresh)
        
        # Display the resulting frame
        cv2.imshow('frame', thresh)
        
        # Show FPS Data
        if debug:
            end = time.time()
            seconds = end - start
            fps  = round((1 / seconds), 1)
            start = time.time()
            print (format(fps))
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else: 
        break
print(i)
# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()

