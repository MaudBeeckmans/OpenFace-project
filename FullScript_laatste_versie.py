# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:12:33 2020

@author: Maud
"""

import numpy as np
import cv2, os, datetime, pandas, time
from psychopy import core, visual, gui
import threading


#%%Variables that should be adapted based on the goals of the experiment
try_out = 1        #0 or 1: 0 then different file per run; 1 then same file overwritten every run 
my_res = '480p'#Select relevant resolution: 480p or 720p

wanted_fps = 15
frames_per_second = wanted_fps

#amount of trials for each condition
n_smile = 10
n_frown = 10

#duration of 1 trial (timing of the video capture)
sec_before_action = 0.5
sec_after_action = 1
sec_action = 1

#Select which webcam you want to use: 0 = implemented webcam; 1 = USB-webcam
webcam_selection = 1

#%% use the defined variables to define the used variables 
n_trials = n_smile + n_frown
video_time = sec_before_action + sec_action + sec_after_action
#n_frames is rounded at 0 decimals to make sure the video-time is as close to possible to the desired video-time
n_frames = int(np.round(video_time * frames_per_second))

wait_time = 1/frames_per_second
actual_video_time = n_frames*wait_time



#%%create a directory to store the video's 
my_home_dir = os.getcwd()

if try_out == 0: 
    # display the gui
    info = { 'Naam': '','Gender': ['man', 'vrouw', 'derde gender'], 'Leeftijd': 0 , 'Nummer': 1}
    already_exists = True 
    while already_exists == True: 
        info_dialogue = gui.DlgFromDict(dictionary=info, title='Information')
        number = info['Nummer']
        my_directory = my_home_dir + '/video' + str(number)
        if not os.path.isdir(my_directory): 
            os.mkdir(my_directory)
            already_exists = False
        else: 
            gui2 = gui.Dlg(title = 'Error')
            gui2.addText("Try another number")
            gui2.show()
else: 
    my_directory = my_home_dir + '/video' + 'test'
    if not os.path.isdir(my_directory): 
        os.mkdir(my_directory)
os.chdir(my_directory)

#%%Create functions  to easlily adapt the resolution and output file

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
        width,height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height

VIDEO_TYPE = {'avi': cv2.VideoWriter_fourcc(*'XVID'),'mp4': cv2.VideoWriter_fourcc(*'XVID')}

#datafilename should include '.mp4' or '.avi' to create that type of output file
def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

def test_camera_position(webcam = 0): 
    cam = cv2.VideoCapture(webcam, cv2.CAP_DSHOW) #CAP_DSHOW toegevoegd zodat camera direct begint met opnemen
    while(True): 
        ret, frame = cam.read() 
        cv2.imshow('frame', frame) 
        # the 'q' button is set as the quitting button 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cam.release()
    cv2.destroyAllWindows()


def check_double_frames(array = None): 
    prev_frame = None
    count = 0
    double = 0
    for frame in array: 
        if np.all(frame == prev_frame): 
            print('repeated frames {}'.format(count))
            double += 1
        prev_frame = frame
        count += 1
    print('Amount of repeated frames: {}'.format(double))

def estimate_fps(webcam = 0,wanted_fps = 15):
    import time
    cap = cv2.VideoCapture(webcam, cv2.CAP_DSHOW) 
    get_dims(cap, res = my_res)
    #cap.set(cv2.CAP_PROP_FPS, wanted_fps)
    num_frames = 100;
    print("Capturing {0} frames".format(num_frames))
    start = time.time()
    for i in range(0, num_frames) :
        ret, frame = cap.read()
    end = time.time()
    seconds = end - start
    fps  = num_frames / seconds;
    print("Estimated frames per second : {0}".format(fps))
    return fps

estimated_fps = estimate_fps(webcam = webcam_selection, wanted_fps=wanted_fps)

# here maybe do a check if estimated_fps is close to wanted_fps, if not stop the program or give an error...

print("FPS DEVIATION : {0}".format(estimated_fps-wanted_fps))

if estimated_fps - wanted_fps < 0: 
    print('Camera cannot sample fast enough')
    core.quit()

#%% define the arrays to store relevant info in etc. 
width, height = STD_DIMENSIONS[my_res]
store_cap_start = np.empty([n_frames, n_trials])
store_cap_end = np.empty([n_frames, n_trials])


store_appearT = np.empty(n_trials)
store_disappearT = np.empty(n_trials)

# store_frames = np.empty([n_trials, n_frames, 480, 640, 3])

#%%Define the window & stimuli for each trial 
win = visual.Window((600, 400), monitor = 'Laptop')

fix = visual.TextStim(win, text = 'x')
smile = visual.TextStim(win, text = 'Smile')
frown = visual.TextStim(win, text = 'Frown')

type_options = np.array(['smile', 'frown'])
type_array = np.concatenate([np.repeat('smile', n_smile), np.repeat('frown', n_frown)])
np.random.shuffle(type_array)

test_camera_position(webcam = webcam_selection)

timer = core.Clock()

cap = cv2.VideoCapture(webcam_selection, cv2.CAP_DSHOW)

for trial in range(n_trials):
    
    this_type = type_array[trial]
    
    appeared = False
    disappeared = False 
    
    filename = 'video-uitproberen' + str(this_type) + str(trial) + '.avi'
    out = cv2.VideoWriter(filename, get_video_type(filename), frames_per_second, get_dims(cap, my_res))
    time.sleep(1)
    fix.draw()
    win.flip()
    
    
    timer.reset()
    for frame_count in range(n_frames): 
        while timer.getTime() < wait_time * frame_count: 
            if timer.getTime() >= 0.5 and appeared == False: 
                if this_type == 'smile': 
                    smile.draw()
                else: 
                    frown.draw()
                win.flip()
                appearT = timer.getTime()
                appeared = True
            elif timer.getTime() >= 1.5 and disappeared == False: 
                win.flip()
                disappearT = timer.getTime()
                disappeared = True
        start_capture = timer.getTime()
        ret, frame = cap.read()
        end_capture = timer.getTime()
        out.write(frame)
        store_cap_start[frame_count, trial] = start_capture
        store_cap_end[frame_count, trial] = end_capture
        # store_frames[trial, frame_count] = frame
    store_appearT[trial] = appearT
    store_disappearT[trial] = disappearT
    out.release()
    print('Action started at time {0}.'.format(appearT))
    print('Action ended at time {0}.'.format(disappearT))
    
    
    
    
    

win.close()
cap.release()


#%%Create a file with all the info stored in it 
trial_count = np.repeat(np.arange(n_trials), n_frames).reshape(n_frames*n_trials)
trial_type = np.repeat(type_array, n_frames).reshape(n_frames*n_trials)

frame_count = np.tile(np.arange(n_frames), n_trials)

start_cap = store_cap_start.reshape(n_frames*n_trials, order = 'F')
end_cap = store_cap_end.reshape(n_frames*n_trials, order = 'F')

action_started = np.repeat(store_appearT, n_frames)
action_ended = np.repeat(store_disappearT, n_frames)

goal_time = np.arange(0, n_frames, 1)
goal_time = goal_time * wait_time
goal_time = np.tile(goal_time, n_trials)

compare_time = start_cap - goal_time

big_array = np.column_stack([trial_count, trial_type, frame_count, start_cap, goal_time, compare_time, end_cap, 
                             action_started, action_ended])
big_DF = pandas.DataFrame.from_records(big_array)
big_DF.columns = ['Trial_number', 'Type', 'frame_count', 'reading_started', 'wanted_start', 'deviation_wanted', 
                  'reading_ended', 'action_started', 'action_ended']
if try_out == 0: 
    DF_file = 'Stored_info' + str(number) + '.csv'
else: 
    DF_file = 'Stored_info_test.csv'
big_DF.to_csv(DF_file, index = False)







