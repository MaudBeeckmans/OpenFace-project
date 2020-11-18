# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:27:15 2020

@author: Maud
"""


import threading 
from psychopy import visual, core
import numpy as np 
import cv2, time, os

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

VIDEO_TYPE = {'avi': cv2.VideoWriter_fourcc(*'DIVX'),'mp4': cv2.VideoWriter_fourcc(*'DIVX')}
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

#datafilename should include '.mp4' or '.avi' to create that type of output file
def get_video_type(filename = None):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']



def check_double_frames(array = None): 
    prev_frame = None
    count = 0
    for frame in array: 
        if np.all(frame == prev_frame): 
            print('repeated frames {}'.format(count))
        prev_frame = frame
        count += 1

def read_camera(frame_count): 
    ret, frame = cap.read()
    store_frame[frame_count] = frame
    out.write(frame)
    

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

my_res = '480p'
filename = 'video-test' + my_res + '.avi'
frames_per_second = 30 #heb ik van de test in file 'calculate_fps_camera')
out = cv2.VideoWriter(filename, get_video_type(filename), frames_per_second, get_dims(cap, my_res))

width, height = STD_DIMENSIONS[my_res]
store_frame = np.empty([60, height, width, 3])

ret, frame = cap.read()
print(frame.shape) #480, 640, 3
time.sleep(2)
threads = []
start = time.perf_counter()
for i in range(60): 
    core.wait(1/frames_per_second)
    t = threading.Thread(target = read_camera, args = [i])
    t.start()
    threads.append(t)
for thread in threads: 
    thread.join()

finish = time.perf_counter()

time_passed = finish-start
print('Finished in {} seconds'.format(time_passed))

cap.release()
out.release()
