# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:45:34 2020

@author: Maud
"""

"""Instructies: 
    Eerst zal een test-scherm verschijnen, kijk hier of je camera goed gericht staat op je gezicht.
    Druk op 'q' wnr dit het geval is.
    Vervolgens begint het echte opname-deel. Vanaf 'smile' of 'frown' op het scherm verschijnt, 
    probeer dit uit te voeren. Doe dit zolang het woord op het scherm blijft staan. 
"""

#versions packages & python: 
    # python: 3.8.6
    # numpy: '1.19.2'
    # cv2: '4.5.0'
    # pandas: '1.1.3'
    # psychopy: 2020.2.5
    
    



#script: camera 2s laten opnemen met fixatiekruis, dan 3s 'smile', dan nog 2s opnemen 

import numpy as np
import cv2, os, datetime, pandas
from psychopy import core, visual, gui


#create a directory to store the video's 
my_home_dir = os.getcwd()
    
# display the gui
info = { 'Naam': '','Gender': ['man', 'vrouw', 'derde gender'], 'Leeftijd': 0 , 'Nummer': 1}
already_exists = True 
while already_exists == True: 
    info_dialogue = gui.DlgFromDict(dictionary=info, title='Information')
    number = info['Nummer']
    my_directory = my_home_dir + '/video' + str(number)
    if not os.path.isdir(my_directory): 
        os.mkdir(my_directory)
        os.chdir(my_directory)
        already_exists = False
    else: 
        gui2 = gui.Dlg(title = 'Error')
        gui2.addText("Try another number")
        gui2.show()




#define some properties of our recording 
#frames_per_second = 30 #heb ik van de test in file 'calculate_fps_camera')
frames_per_second = 31 #met dit script kan webcam samplen aan 31 frames per seconde
                        # als je met deze timing & resolutie van 480p werkt toch precies
my_res = '480p'

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
VIDEO_TYPE = {'avi': cv2.VideoWriter_fourcc(*'DIVX'),'mp4': cv2.VideoWriter_fourcc(*'DIVX')}
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

#datafilename should include '.mp4' or '.avi' to create that type of output file
def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

def test_camera_position(): 
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #CAP_DSHOW toegevoegd zodat camera direct begint met opnemen
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



#%%

sec_before_action = 1
sec_after_action = 1
sec_action = 1

video_time = sec_before_action + sec_action + sec_after_action

win = visual.Window((600, 400), monitor = 'Laptop')

fix = visual.TextStim(win, text = 'x')
smile = visual.TextStim(win, text = 'Smile')
frown = visual.TextStim(win, text = 'Frown')


type_options = np.array(['smile', 'frown'])
n_smile = 2
n_frown = 2
n_trials = n_smile + n_frown
#type_array_binary = np.concatenate([np.zeros(n_smile), np.ones(n_frown)])
type_array = np.concatenate([np.repeat('smile', n_smile), np.repeat('frown', n_frown)])
#shuffle the array to have random sequence of the trials
#np.random.shuffle(type_array_binary)
np.random.shuffle(type_array)

actionstart_frame = np.empty(n_trials)
actionstart_time = np.empty(n_trials)
actionend_frame = np.empty(n_trials)
actionend_time = np.empty(n_trials)

test_camera_position()

sec_before_action = 1
sec_after_action = 1
sec_action = 1

video_time = sec_before_action + sec_action + sec_after_action
timer = core.Clock()

for trial in range(n_smile + n_frown): 
    frame_count = 1
    #create the possibility to capture video 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
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
    #record video for certain time: 
    while timer.getTime() < video_time: 
        ret, frame = cap.read()
        #timepoint = cap.get(cv2.CAP_PROP_FPS)#werkt niet wnr met webcam werken 
        out.write(frame)
        frame_count += 1
        if frame_count == int(frames_per_second*sec_before_action): 
            if this_type == 'smile': 
                smile.draw()
            else: 
                frown.draw()
            win.flip()
            startT = timer.getTime()
            #from the next frame onwards pp. has been instructed to do the action 
            startF = frame_count + 1
        elif frame_count == int(frames_per_second*(sec_before_action + sec_action)):
            # fix.draw()
            win.flip()
            endT = timer.getTime()
            #from the next frame onwards, pp. won't have to do the action anymore
            endF = frame_count + 1
        #cv2.imshow('frame',frame)
    end = timer.getTime()
    print(end)
    print(frame_count-1)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    actionstart_time[trial] = startT
    actionstart_frame[trial] = startF  #from this frame onwards the action was shown the first time
    actionend_time[trial] = endT
    actionend_frame[trial] = endF  #from this frame onwards the action cue was gone already
    print('Action started at time {0}, frame {1}.'.format(startT, startF))
    print('Action ended at time {0}, frame {1}.'.format(endT, endF))
win.close()


big_array = np.column_stack([type_array, actionstart_time, actionstart_frame, actionend_time, 
                             actionend_frame])
big_DF = pandas.DataFrame.from_records(big_array)
big_DF.columns = ['Type', 'action started ms', 'action started F', 'action ended ms', 'action ended F']
DF_file = 'Stored_info.csv'
big_DF.to_csv(DF_file, index = True)

print(actionstart_time)
print(actionstart_frame)
print(actionend_time)
print(actionend_frame)

core.quit()
#recor video untill you quit, pressing 'q'
# while True:
#     ret, frame = cap.read()
#     out.write(frame)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break



#compare output with and without waiting 2s before starting to record the video 
#withouth waiting: 
    # time after 31 frames recorded: [1.1645041 1.1994791 1.1836837 1.2001949]
    # [32. 32. 32. 32.]
    # time after 93 frames recorded: [3.2128685 3.233072  3.2164134 3.2332386]
    # [94. 94. 94. 94.]
    # total frames recorded : 117
#with waiting 2 seconds: 
    # time after 31 frames recorded: [0.99583   0.9830465 0.9853312 0.9643111]
    # [32. 32. 32. 32.]
    # time after 93 frames recorded[3.0298337 3.0335942 3.0329871 3.0141079]
    # [94. 94. 94. 94.]
    #total frames recorded: 124





