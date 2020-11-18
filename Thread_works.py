# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:12:56 2020

@author: Maud
"""

from psychopy import core, visual
import numpy as np 
import cv2, time
import threading 

def do_something(second): 
    print('Started {}'.format(second))
    time.sleep(second)
    print('Ended {}'.format(second))


    

threads = []

start = time.perf_counter()

for i in range(120): 
    time.sleep(0.03333)
    t = threading.Thread(target = do_something, args = [1])
    t.start()
    threads.append(t)
for thread in threads: 
    thread.join()
    
    
finish = time.perf_counter()

time_passed = finish-start
print('Finished in {} seconds'.format(time_passed))