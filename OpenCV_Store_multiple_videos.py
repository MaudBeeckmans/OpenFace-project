# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:37:02 2020

@author: Maud
"""

"""Instructies: 
    Eerst zal een test-scherm verschijnen, kijk hier of je camera goed gericht staat op je gezicht.
    Druk op 'q' wnr dit het geval is.
    Vervolgens begint het echte opname-deel. Vanaf 'smile' of 'frown' op het scherm verschijnt, 
    probeer dit uit te voeren. Doe dit zolang het woord op het scherm blijft staan. 
"""

"""
Verschillende zaken aangepast: 
    * timings van elk frame worden mee bijgehouden in csv file 
    * 1ste frames worden niet opgenomen: hadden enorm kleine tussentijd, dus cut-off gezet voor minimum 
    tijd tussen de eerste frames
       - gebruikte cut-offs: 0.025 voor 480p & 0.08 voor 720p
Verschillende zaken uitgetest: 
    * 480p: daglicht vs. kunstlicht vs. vrij donker
       - vrij donker: 1frame duurt gemiddeld 0.6s (= 16,666 fps)
       - daglicht & kunstlicht: 1frame duurt gemiddeld 0.033s (= 30fps)
         --> steeds ca. dezelfde frames die wel langer duren (ca. 0.048s: frames 10, 24, 38, 52)
         --> timings lijken wel redelijk consistent over trials heen 
    * 720p: zelfde resultaten bij kunstlicht & vrij donker
       - 1frame duurt gemiddeld 0.099s (ca. 10fps)
       - alle frames wel redelijk consistent qua timing [0.09; 0.12[
    * 1080p: werkt niet, 'opgeslagen video's' kan ik niet afspelen 
==> voor goede timing best met 480p werken, misschien nog uittesten of een webcam betere resultaten kan halen
"""

#versions packages & python:  (Access via e.g. np.__version__)
    # python: 3.6.11
    # numpy: '1.19.2'
    # cv2: '4.5.0'
    # pandas: '1.1.4'
    #psychopy: 2020.2.5



import numpy as np
import cv2, os, datetime, pandas
from psychopy import core, visual, gui

#%%Variables that should be adapted based on the goals of the experiment
try_out = 1         #0 or 1: 0 then different file per run; 1 then same file overwritten every run 
my_res = '720p'#Select relevant resolution: 480p or 720p

#amount of trials for each condition
n_smile = 2
n_frown = 2

#duration of 1 trial (timing of the video capture)
sec_before_action = 0.5
sec_after_action = 1
sec_action = 1

#Select which webcam you want to use: 0 = implemented webcam; 1 = USB-webcam
webcam_selection = 1


if my_res == '480p': 
    cutoff = 0.028
    frames_per_second = 30
else: 
    cutoff = 0.08
    frames_per_second  = 10

#%%
#create a directory to store the video's 
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
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

#datafilename should include '.mp4' or '.avi' to create that type of output file
def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

def test_camera_position(webcam = 0): 
    cam = cv2.VideoCapture(webcam, cv2.CAP_DSHOW) #CAP_DSHOW toegevoegd zodat camera direct begint met opnemen
    while(True): 
        # Capture the video frame by frame 
        ret, frame = cam.read() 
        # Display the resulting frame 
        cv2.imshow('frame', frame) 
        # the 'q' button is set as the quitting button 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cam.release()
    cv2.destroyAllWindows()

def check_double_frames(array = None): 
    prev_frame = None
    count = 0
    for frame in array: 
        if np.all(frame == prev_frame): 
            print('repeated frames {}'.format(count))
        prev_frame = frame
        count += 1


#%%



video_time = sec_before_action + sec_action + sec_after_action

win = visual.Window((600, 400), monitor = 'Laptop')

fix = visual.TextStim(win, text = 'x')
smile = visual.TextStim(win, text = 'Smile')
frown = visual.TextStim(win, text = 'Frown')


type_options = np.array(['smile', 'frown'])
n_trials = n_smile + n_frown
#type_array_binary = np.concatenate([np.zeros(n_smile), np.ones(n_frown)])
type_array = np.concatenate([np.repeat('smile', n_smile), np.repeat('frown', n_frown)])
#shuffle the array to have random sequence of the trials
#np.random.shuffle(type_array_binary)
np.random.shuffle(type_array)


actionstart_time = np.empty(n_trials)
actionend_time = np.empty(n_trials)


test_camera_position(webcam = webcam_selection)

sec_before_action = 0.5
sec_after_action = 0.5
sec_action = 1

video_time = sec_before_action + sec_action + sec_after_action
n_frames = int(video_time*frames_per_second)
timer = core.Clock()
timer2 = core.Clock()

store_frame = np.empty([n_frames, 480, 640, 3])

store_betweentime = np.empty([n_frames, n_trials]) #for every trial its own column

for trial in range(n_smile + n_frown): 
    this_frame = 0
    #create the possibility to capture video 
    cap = cv2.VideoCapture(webcam_selection, cv2.CAP_DSHOW)
    i = 0
    #select_type = int(type_array_binary[trial])
    #this_type = type_options[select_type]
    this_type = type_array[trial]
    #create output possibility 
    filename = 'video-uitproberen' + str(this_type) + str(trial) + '.avi'
    out = cv2.VideoWriter(filename, get_video_type(filename), frames_per_second, get_dims(cap, my_res))
    fix.draw()
    win.flip()
    core.wait(1)  #seems necessary to have valid timing, maybe camera !! to adapt a bit first 2 sec 
    timer.reset()
    timer2.reset()
    #record video for certain time: 
    while this_frame < n_frames: 
        ret, frame = cap.read()
        this_time = timer.getTime()
        timer.reset()
        if this_time < cutoff and this_frame == 0: 
            timer2.reset()
            print(frame.shape)
        else: 
            store_betweentime[this_frame, trial] = this_time
            out.write(frame)
            this_frame += 1
        if this_frame == int(frames_per_second*sec_before_action)-1: 
            if this_type == 'smile': 
                smile.draw()
            else: 
                frown.draw()
            win.flip()
            startT = timer2.getTime()
        elif this_frame == int(frames_per_second*(sec_before_action + sec_action))-1:
            win.flip()
            endT = timer2.getTime()
    end = timer2.getTime()
    print(end)
    print(this_frame-1)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    actionstart_time[trial] = startT
    actionend_time[trial] = endT
    print('Action started at time {0}.'.format(startT))
    print('Action ended at time {0}.'.format(endT))
    #check_double_frames(array = store_frame)
win.close()

#%%
trials_array = np.column_stack([type_array, actionstart_time, actionend_time])
big_array = np.repeat(trials_array, n_frames, axis = 0)
store_betweentime = store_betweentime.T
store_betweentime_all = store_betweentime.reshape(n_frames*n_trials)
frames_counting = np.tile(np.arange(n_frames), n_trials)
big_array = np.column_stack([big_array, store_betweentime_all, frames_counting])
big_DF = pandas.DataFrame.from_records(big_array)
big_DF.columns = ['Type', 'action_started_ms', 'action_ended_ms', 'time_between_each_frame', 'frame_count']
if try_out == 0: 
    DF_file = 'Stored_info' + str(number) + '.csv'
else: 
    DF_file = 'Stored_info_test.csv'
big_DF.to_csv(DF_file, index = False)

print(actionstart_time)
print(actionend_time)

core.quit()
