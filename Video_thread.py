# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:27:15 2020

@author: Maud
"""


import threading 
from psychopy import visual, core
import numpy as np 
import cv2, time


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
    
store_frame = np.empty([60, 480, 640, 3])
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, frame = cap.read()
print(frame.shape) #480, 640, 3
time.sleep(2)
threads = []
start = time.perf_counter()
for i in range(60): 
    core.wait(0.03333)
    t = threading.Thread(target = read_camera, args = [i])
    t.start()
    threads.append(t)
for thread in threads: 
    thread.join()

finish = time.perf_counter()

time_passed = finish-start
print('Finished in {} seconds'.format(time_passed))