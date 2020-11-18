# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:50:29 2020

@author: Maud
"""


import numpy as np
import cv2, os, datetime, pandas
from psychopy import core, visual, gui

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
#adapt the resolution of the video you'll make (video is named cap)
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes: dictionary you can sample from 
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    #if input resolution doesn't match anything of dictionary, then put is to 480p
    width, height = STD_DIMENSIONS["480p"]
    #if input resolution is part of dictionary, use that 
    if res in STD_DIMENSIONS:
        #get correct width & height from the dictionary 
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    #use the function to change the resolution of the video you'll capture
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
#2 types of video's very useful, which used is dependent on the datafile name above
VIDEO_TYPE = {'avi': cv2.VideoWriter_fourcc(*'XVID'),'mp4': cv2.VideoWriter_fourcc(*'XVID')}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


#datafilename should include '.mp4' or '.avi' to create that type of output file
my_res = '720p'
filename = 'video-test' + my_res + '.avi'
frames_per_second = 30 #heb ik van de test in file 'calculate_fps_camera')

video_time = 2
timer = core.Clock()
store_time = np.empty(22)

frame_count = 0
#create the possibility to capture video 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#create output possibility 
out = cv2.VideoWriter(filename, get_video_type(filename), frames_per_second, get_dims(cap, my_res))
core.wait(1)
i = 0
timer.reset()
#record video for certain time: 
while timer.getTime() < video_time: 
    store_time[i] = timer.getTime()
    ret, frame = cap.read()
    out.write(frame)
    frame_count += 1
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1
end = timer.getTime()
print(end)
print(frame_count-1)

cap.release()
out.release()
cv2.destroyAllWindows()
prev_time =0
for time in store_time: 
    print(time-prev_time)
    prev_time = time





























# import numpy as np
# import cv2
# from psychopy import core

# n_frames = 100
# store_between_time = np.empty(n_frames)
# recorded_times = np.empty(n_frames)
# clock = core.Clock()
# clock2 = core.Clock()
# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fps = 28,5
# dims = (640, 480)
# out = cv2.VideoWriter('output.avi',fourcc, fps, dims)
# count = 0
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# while(cap.isOpened() and count < n_frames):
#     if count == 0: 
#         clock.reset()
#         clock2.reset()
#     ret, frame = cap.read()
#     recorded_time = clock2.getTime()
#     between_t = clock.getTime()
#     clock.reset()
#     recorded_times[count] = recorded_time
#     # store_between_time[count] = between_t
#     if ret==True:

#         # write the flipped frame
#         out.write(frame)

#         #cv2.imshow('frame',frame)
#         count += 1
#         print(count)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# total_time = clock.getTime()
# print(total_time)
# # print(store_between_time)
# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()