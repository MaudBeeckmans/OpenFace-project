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
try_out = 1       #0 or 1: 0 then different file per run; 1 then same file overwritten every run 
my_res = '480p'#Select relevant resolution: 480p or 720p

wanted_fps = 15
frames_per_second = wanted_fps
n_frames_estimation = 100

#amount of trials for each condition
n_smile = 2
n_frown = 2

#duration of 1 trial (timing of the video capture)
sec_before_action = 0.5
sec_after_action = 1
sec_action = 1

#Select which webcam you want to use: 0 = implemented webcam; 1 = USB-webcam
webcam_selection = 1

#define the output file of the video
video_type = '.avi'


#%% use the defined variables to define the used variables 
n_trials = n_smile + n_frown
video_time = sec_before_action + sec_action + sec_after_action
#n_frames is rounded at 0 decimals to make sure the video-time is as close to possible to the desired video-time
    #e.g. with video-time of 2.5 & fps = 15, the n_frames would be 37.5, but n_frames has to be an integer, 
    #so we round this to most similar integer 
n_frames = int(np.round(video_time * frames_per_second))

wait_time = 1/frames_per_second
#what actual duration of video will be, since we rounded the n_frames & the n_frames defines the duration of the video
actual_video_time = n_frames*wait_time



#%%create a directory to store the video's 
my_home_dir = os.getcwd()

#create the correct folder to store video's in
    #with try_out = 0: video's are always stored in the same folder (files are overwritten every run)
    #with try_out = 1: each pp. gets his own folder & no pp_numbers can be repeated (files can't be overwritten)
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
    number = 0
os.chdir(my_directory)

#%%Create functions  to easlily adapt the resolution and output file

# Standard Video Dimensions Sizes: dictionary you can sample from 
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
}

# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='480p'):
    #if input resolution doesn't match anything of dictionary, then put is to 480p
    width, height = STD_DIMENSIONS["480p"]
    #if input resolution is part of dictionary, use that 
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    cap.set(3, width)
    cap.set(4, height)
    return width, height



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

# function that can be used when testing the script: to make sure the read frames are all different & no repitions occur 
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

#estimate the framerate of your camera empirically
def estimate_fps(webcam = 0,wanted_fps = 15, num_frames = 100):
    import time
    cap = cv2.VideoCapture(webcam, cv2.CAP_DSHOW) 
    get_dims(cap, res = my_res)
    #cap.set(cv2.CAP_PROP_FPS, wanted_fps)
    print("Capturing {0} frames".format(num_frames))
    start = time.time()
    for i in range(0, num_frames) :
        ret, frame = cap.read()
    end = time.time()
    seconds = end - start
    fps  = num_frames / seconds;
    print("Estimated frames per second : {0}".format(fps))
    return fps

estimated_fps = estimate_fps(webcam = webcam_selection, wanted_fps=wanted_fps, num_frames = n_frames_estimation)

print("FPS DEVIATION : {0}".format(estimated_fps-wanted_fps))

# make sure the script quits when the wanted_fps is too high, when camera cannot generate those fps 
    #script seems to work perfectly when estimated_fps - wanted_fps > 0.5
if estimated_fps - wanted_fps < 0.5: 
    print('Camera cannot sample fast enough')
    core.quit()

#%% define the arrays to store relevant info in etc. 
store_cap_start = np.empty([n_frames, n_trials])
store_cap_end = np.empty([n_frames, n_trials])


store_appearT = np.empty(n_trials)
store_disappearT = np.empty(n_trials)

# store_frames = np.empty([n_trials, n_frames, 480, 640, 3])   #only useful when wanting to chech if there are double frames recorded

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

#Start the actual process of video-capturing 
cap = cv2.VideoCapture(webcam_selection, cv2.CAP_DSHOW)

for trial in range(n_trials):
    
    this_type = type_array[trial]
    
    appeared = False
    disappeared = False 
    
    #create the video-file for this trial
    filename = 'video_pp' + str(number) + str(this_type) + str(trial) + video_type
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), frames_per_second, get_dims(cap, my_res))
    #1 second rest between each trial (useful for participant & necessary for the camera)
    time.sleep(1)
    fix.draw()
    win.flip()
    
    
    timer.reset()
    for frame_count in range(n_frames): 
        #while-loop: manually define when the next frame should be captured: only after the in_between_time has passed
          # with in_between_time = wait_time * frame_count ans with wait_count = 1/wanted_fps 
        #note: if wait_time * frame_count = sec_before_action or (sec_before_action + sec_action) at a certain frame, 
                #then the cap.read will be delayed ca. 16ms since the window is first flipped 
                #happens e.g. when working with fps = 20
        while timer.getTime() < wait_time * frame_count: 
            #if & elif: check time-based whether the stimuli should appear or disappear while waiting for the next frame to be captured
            if timer.getTime() >= sec_before_action and appeared == False: 
                #if - elif: define which stimulus should appear on screen (based on the condition we're in)
                if this_type == 'smile': 
                    smile.draw()
                else: 
                    frown.draw()
                win.flip()
                #measure appearT after flip to be sure the stimulus is visible at this moment
                appearT = timer.getTime()
                appeared = True
            elif timer.getTime() >= (sec_before_action + sec_action) and disappeared == False: 
                win.flip()
                #measure disappearT after flip to be sure the stimulus is not visible anymore at this moment 
                disappearT = timer.getTime()
                disappeared = True
        #start the capture & store the frame after termination of the while-loop
        #start_capture: store when exactly the camera has started capturing this frame
        start_capture = timer.getTime()
        ret, frame = cap.read()
        out.write(frame)        
        #end_capture: store when the camera is done storing & writing the frame 
        end_capture = timer.getTime()
        #store relevant variables in an array 
        store_cap_start[frame_count, trial] = start_capture
        store_cap_end[frame_count, trial] = end_capture
        # store_frames[trial, frame_count] = frame  #can be used to check whether the same frame is recorded twice, 
                                #this in combination with the check_double_frames function after collecting all frames
    
    store_appearT[trial] = appearT
    store_disappearT[trial] = disappearT
    #release the output file for this trial 
    out.release()
    print('Action started at time {0}.'.format(appearT))
    print('Action ended at time {0}.'.format(disappearT))
    
    

win.close()
cap.release()


#%%Create a file with all the info stored in it 
#add a column with the number of the trial, one with the type of the condition & one with the number of the frame 
trial_count = np.repeat(np.arange(n_trials), n_frames).reshape(n_frames*n_trials)
trial_type = np.repeat(type_array, n_frames).reshape(n_frames*n_trials)
frame_count = np.tile(np.arange(n_frames), n_trials)

#reshape arrays to be able to store them in csv 
start_cap = store_cap_start.reshape(n_frames*n_trials, order = 'F')
end_cap = store_cap_end.reshape(n_frames*n_trials, order = 'F')

action_started = np.repeat(store_appearT, n_frames)
action_ended = np.repeat(store_disappearT, n_frames)

#make an array with the 'desired start_reading times' (based on mathematical function)
goal_time = np.arange(0, n_frames, 1)
goal_time = goal_time * wait_time
goal_time = np.tile(goal_time, n_trials)

#compare the 'desired start_reading times (goal_time)' with the 'actual_start_reading_time (start_cap)'
compare_time = start_cap - goal_time

#combine all above arrays into 1 big array and save this in a csv file 
big_array = np.column_stack([trial_count, trial_type, frame_count, start_cap, goal_time, compare_time, end_cap, 
                             action_started, action_ended])
big_DF = pandas.DataFrame.from_records(big_array)
big_DF.columns = ['Trial_number', 'Condition', 'Frame_count', 'Reading_started', 'Wanted_start', 'Deviation_wanted', 
                  'Reading_ended', 'Action_started', 'Action_ended']
if try_out == 0: 
    DF_file = 'Stored_info' + str(number) + '.csv'
else: 
    DF_file = 'Stored_info_test.csv'
big_DF.to_csv(DF_file, index = False)
