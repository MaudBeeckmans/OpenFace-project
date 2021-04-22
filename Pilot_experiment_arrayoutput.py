# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 22:16:50 2021

@author: Maud
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:11:22 2021

@author: Maud
"""

"""
* When actually doing the experiment: 
    - fullscreen = True
    - used_monitor should be adapted
    - webcam_selection = 1
* OASIS pictures should be in a subfolder called 'Images' within the same folder as the code. 
   - so e.g. file stored in "C://Users/experiment"
   - then images stored in "C://Users/experiment/Images"
* Question: Should we ask participants to fixate or not? As we might be interested in the eye gaze as well.
"""

import numpy as np
import cv2, os, time
import pandas as pd
from psychopy import core, visual, gui, event, data




#%%Variables that should be adapted based on the goals of the experiment
try_out = 0       #0 or 1: 0 then different file per run; 1 then same file overwritten every run 
my_res = '480p'#Select relevant resolution: 480p or 720p

fullscreen = False
used_monitor = 'Laptop'

wanted_fps = 15
frames_per_second = wanted_fps
n_frames_estimation = 100

#duration of 1 trial (timing of the video capture)
sec_fix = 1
sec_stim = 2
sec_fb = 1
sec_blank = 1

#stimuli
n_blocks = 3

n_per_cond_block = 75
n_pos = n_per_cond_block*3
n_neg = n_per_cond_block*3
n_trials = n_pos + n_neg
n_blocktrials = n_per_cond_block * 2

#Select which webcam you want to use: 0 = implemented webcam; 1 = USB-webcam
webcam_selection = 0

#define the output file of the video
video_type = '.avi'


#%% use the defined variables to define the used variables 
video_time = sec_fix + sec_stim + sec_fb
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
my_output_dir = os.path.join(my_home_dir, 'Videos_pilot')
if not os.path.isdir(my_output_dir): 
    os.mkdir(my_output_dir)
if try_out == 0: 
    # display the gui
    info = { 'Naam': '','Gender': ['man', 'vrouw', 'derde gender'], 'Leeftijd': 0 , 'Nummer': 1}
    already_exists = True 
    while already_exists == True: 
        info_dialogue = gui.DlgFromDict(dictionary=info, title='Information')
        number = info['Nummer']
        video_directory = os.path.join(my_output_dir, str('Participant' + str(number)))
        if not os.path.isdir(video_directory): 
            os.mkdir(video_directory)
            already_exists = False
        else: 
            gui2 = gui.Dlg(title = 'Error')
            gui2.addText("Try another number")
            gui2.show()
else: 
    video_directory = my_output_dir
    number = 0
# os.chdir(my_directory)

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

test_camera_position(webcam = webcam_selection)

#%%select pictures from the OASIS database
file = os.path.join(os.getcwd(), "OASIS.csv")
OASIS_df = pd.read_csv(file, sep = ',')
OASIS_sorted = OASIS_df.sort_values(by=['Valence_mean'])
OASIS_sorted_np = OASIS_sorted.to_numpy()

OASIS_columns = OASIS_df.columns

neg_pics = OASIS_sorted_np[0:n_neg, :]
pos_pics = OASIS_sorted_np[-n_pos::, :]

#randomly shuffle the positive and negative pictures 
np.random.shuffle(neg_pics)
np.random.shuffle(pos_pics)

#add column with 'positive' or 'negative' for easy feedback generation
neg_array = np.repeat('negative', n_pos)
pos_array = np.repeat('positive', n_neg)
neg_pics = np.column_stack([neg_pics, neg_array])
pos_pics = np.column_stack([pos_pics, pos_array])

#%%create the relevant stimuli for all trials
if fullscreen == True: 
    win = visual.Window(fullscr = True, monitor = used_monitor, units = 'deg')
else: 
    win = visual.Window((800, 600), monitor = used_monitor, units = 'deg')

fix = visual.TextStim(win, text = '+')
circle = visual.Circle(win, radius = 0.5, lineColor = 'black')
feedback = visual.TextStim(win, text = '', color = 'white')

all_color_options = np.array([['orange', 'blue'], ['pink', 'green'], ['brown', 'yellow']])
resp_options = np.array(['s', 'l', 'esc', 'escape'])

correct_dimensions = ['color', 'affect', 'affect']

#%%template for the instructions
message_template = visual.TextStim(win, text = '')

def message(message_text = '', duration = 0, response_keys = ['space'], color = 'white', height = 0.1, 
            wrapWidth = 1.9, flip = True, position = (0,0), speedy = 0, align = 'center'):
    message_template.text = message_text
    message_template.pos = position
    message_template.units = "norm"
    message_template.color = color
    message_template.height = height 
    message_template.wrapWidth = wrapWidth 
    message_template.alignText = align
    message_template.draw()
    if flip == True: 
        win.flip()
        if speedy == 1: 
            core.wait(0.01)
        else: 
            if duration == 0:
                #when duration = 0, wait till participant presses the right key (keys allowed can be found in response_keys, default allowed key is 'space')
                event.waitKeys(keyList = response_keys)
            else: 
                core.wait(duration)


#%%Define the instructions
instructions1 = str('In dit blok zal er telkens een cirkel in het midden van het scherm verschijnen.'
                    + ' Als deze cirkel ORANJE is druk \'S\'; als deze BLAUW is, druk \'L\'.'
                    + ' De foto\'s die verschijnen zijn niet van belang voor de taak.'
                    + ' Probeer zo SNEL en zo ACCURAAT mogelijk te antwoorden.')
instructions2 = str('In dit blok zal er telkens een foto in het midden van het scherm verschijnen.'
                    + ' Als deze foto eerder NIET LEUK is druk \'S\'; als deze eerder LEUK is, druk \'L\'.'
                    + ' De cirkels die ook in het midden verschijnen zijn niet van belang voor de taak.'
                    + ' Probeer zo SNEL en zo ACCURAAT mogelijk te antwoorden.')
instructions3a = str('Dit blok is gelijkaardig aan vorig blok.'
                    + ' Er zal er weer telkens een foto in het midden van het scherm verschijnen.'
                    + ' Als deze foto eerder NIET LEUK is druk \'S\'; als deze eerder LEUK is, druk \'L\'.'
                    + ' De cirkels die ook in het midden verschijnen zijn niet val belang voor de taak.'
                    + ' Probeer zo SNEL en zo ACCURAAT mogelijk te antwoorden.')
instructions3b = str('Wat extra is in dit blok is dat je elke trial ook met een gezichtsuitdrukking gaat antwoorden.'
                     + ' Als de foto NIET LEUK is, frons dan; als de foto LEUK is, smile dan.'
                     + ' Probeer dit te doen zolang de foto op het scherm verschijnt.'
                     + ' Dit lijkt misschien wat raar, maar het is toch heel belangrijk dat je dit zo goed mogelijk doet.')


all_instructions = np.array([instructions1, instructions2, instructions3a, instructions3b])

#%% define the arrays to store relevant info in etc. 
store_cap_start = np.empty([n_blocks, n_blocktrials, n_frames])
store_cap_end = np.empty([n_blocks, n_blocktrials, n_frames])


store_appearT = np.empty([n_blocks, n_blocktrials])
store_disappearT = np.empty([n_blocks, n_blocktrials])

all_pics = np.empty([n_trials, pos_pics.shape[1]])
all_colors = np.empty([n_trials, pos_pics.shape[1]])
all_accuracies = np.empty([n_blocks, n_blocktrials])
all_corResps = np.empty([n_blocks, n_blocktrials])
all_RTs= np.empty([n_blocks, n_blocktrials])
#%%Define the window & stimuli for each trial 


timer = core.Clock()
RT_clock = core.Clock()

#Start the actual process of video-capturing 
cap = cv2.VideoCapture(webcam_selection, cv2.CAP_DSHOW)

#info.pop('Naam')
output_file = os.path.join(video_directory, str("output_participant" + str(number) + '.csv'))

spatie = 'Druk op spatie om verder te gaan'
name = 'Maud'
message(message_text = 'Dag ' + name, duration = 1)
win.flip()


for block in range(n_blocks): 
    Quit = False
    instructions = all_instructions[block]
    message(message_text = instructions, align = 'left')
    win.flip()
    if block == 2: 
        message(message_text = instructions3b, align = 'left')
        win.flip()
    block_directory = os.path.join(video_directory, str('block' + str(block+1)))
    if not os.path.isdir(block_directory): 
            os.mkdir(block_directory)
    color_options = all_color_options[block, :]
    affect_options = np.array(['negative', 'positive'])
    block_colors = np.repeat(all_color_options[block, :], n_per_cond_block)
    block_cond_selection = (block*n_per_cond_block, (block+1)*n_per_cond_block)
    block_selection = (block*n_blocktrials, (block+1)*n_blocktrials)
    block_pos_pics = pos_pics[block_cond_selection[0]:block_cond_selection[1], :]
    block_neg_pics = neg_pics[block_cond_selection[0]:block_cond_selection[1], :]
    block_pics = np.row_stack([block_pos_pics, block_neg_pics])
    np.random.shuffle(block_colors)
    np.random.shuffle(block_pics)
    
    correct_dim = correct_dimensions[block]
    if correct_dim == "color": 
        corResp_array = (block_colors == color_options[1])*1
    elif correct_dim == "affect": 
        corResp_array = (block_pics[:, 10]== affect_options[1])*1
    
    if block == 0: 
        all_pics = block_pics
        all_colors = block_colors
        all_corResps = corResp_array
    else: 
        all_pics = np.row_stack([all_pics, block_pics])
        all_colors = np.row_stack([all_colors, block_colors])
        all_corResps = np.row_stack([all_corResps, corResp_array])
    # all_pics[block_selection[0]:block_selection[1], :] = block_pics
    # all_colors[block_selection[0]:block_selection[1]] = block_colors
    # all_corResps[block_selection[0]:block_selection[1]] = corResp_array    
    
    
    
    for trial in range(n_blocktrials):
        this_type = block_pics[trial, 10]
        this_pic_name = os.path.join(my_home_dir, 'Images', str(block_pics[trial, 1] + '.jpg'))
        this_pic = visual.ImageStim(win, image = this_pic_name, units = 'deg', size = (8, 8))
        circle.fillColor = block_colors[trial]
        
        
        correct_resp = resp_options[corResp_array[trial]]
            
        
        appeared = False
        disappeared = False 
        feedback_on_screen = False
        
        #create the video-file for this trial
        filename = 'pp' + str(number) + '_block' + str(block+1) + '_trial' + str(trial) + '_' + str(this_type) + video_type
        out = cv2.VideoWriter(os.path.join(block_directory, filename), cv2.VideoWriter_fourcc(*'XVID'), frames_per_second, get_dims(cap, my_res))
        #1 second rest between each trial (useful for participant & necessary for the camera)
        win.flip()
        time.sleep(sec_blank)
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
                if timer.getTime() >= sec_fix and timer.getTime() < (sec_fix + sec_stim):
                    if appeared == False: 
                        #if - elif: define which stimulus should appear on screen (based on the condition we're in)
                        this_pic.draw()
                        circle.draw()
                        win.flip()
                        RT_clock.reset()
                        #measure appearT after flip to be sure the stimulus is visible at this moment
                        appearT = timer.getTime()
                        response = event.getKeys(keyList = resp_options, timeStamped = RT_clock)
                        appeared = True
                    elif response == []: 
                        response = event.getKeys(keyList = resp_options, timeStamped = RT_clock)
                elif timer.getTime() >= (sec_fix + sec_stim):
                    if disappeared == False: 
                        win.flip()
                        #measure disappearT after flip to be sure the stimulus is not visible anymore at this moment 
                        disappearT = timer.getTime()
                        disappeared = True
                    elif feedback_on_screen == False: 
                        if response == []: 
                            feedback_text = 'Te traag'    
                            accuracy = -1
                            response = [None, None]
                        elif np.array(response).squeeze()[0] == 'esc' or np.array(response).squeeze()[0] == 'escape': 
                            Quit = True
                            break
                        elif np.array(response).squeeze()[0] == correct_resp: 
                            feedback_text = 'Juist'
                            accuracy = 1
                            response = np.array(response).squeeze()
                        else: 
                            feedback_text = 'Fout'
                            accuracy = 0
                            response = np.array(response).squeeze()
                        feedback.text = feedback_text
                        feedback.draw()
                        win.flip()
                        feedback_on_screen = True
            #start the capture & store the frame after termination of the while-loop
            #start_capture: store when exactly the camera has started capturing this frame
            start_capture = timer.getTime()
            ret, frame = cap.read()
            out.write(frame)        
            #end_capture: store when the camera is done storing & writing the frame 
            end_capture = timer.getTime()
            #store relevant variables in an array 
            store_cap_start[block, trial, frame_count] = start_capture
            store_cap_end[block, trial, frame_count] = end_capture
            if Quit == True: 
                break
        if Quit == True: 
                break    
        store_appearT[block, trial] = appearT
        store_disappearT[block, trial] = disappearT
        all_accuracies[block, trial] = accuracy
        all_RTs[block, trial] = response[1]
        #release the output file for this trial 
        out.release()
        
        
message(message_text = 'Dit is het einde van het experiment. Bedankt voor je deelname.')        
    
win.close()
cap.release()
    

#%%Create a file with all the info stored in it 
#add a column with the number of the trial, one with the type of the condition & one with the number of the frame 
trial_count = np.tile(np.repeat(np.arange(n_blocktrials), n_frames), n_blocks)
block_count = np.repeat(np.arange(n_blocks), n_frames*n_blocktrials).reshape(n_frames*n_trials)
# trial_type = np.repeat(type_array, n_frames).reshape(n_frames*n_trials)
frame_count = np.tile(np.arange(n_frames), n_trials)

#reshape arrays to be able to store them in csv 
start_cap = store_cap_start.reshape(n_frames*n_trials, order = 'C')
end_cap = store_cap_end.reshape(n_frames*n_trials, order = 'C')

store_pics = np.repeat(all_pics, n_frames, axis = 0)
store_colors = np.repeat(all_colors.reshape(n_trials, order = 'C'), n_frames)
store_accuracies = np.repeat(all_accuracies.reshape(n_trials, order = 'C'), n_frames)
store_corResps = np.repeat(all_corResps.reshape(n_trials, order = 'C'), n_frames)
store_RTs = np.repeat(all_RTs.reshape(n_trials, order = 'C'), n_frames)

action_started = np.repeat(store_appearT.reshape(n_trials), n_frames)
action_ended = np.repeat(store_disappearT.reshape(n_trials), n_frames)

# #make an array with the 'desired start_reading times' (based on mathematical function)
goal_time = np.arange(0, n_frames, 1)
goal_time = goal_time * wait_time
goal_time = np.tile(goal_time, n_blocktrials)
goal_time = np.tile(goal_time, n_blocks)

#compare the 'desired start_reading times (goal_time)' with the 'actual_start_reading_time (start_cap)'
compare_time = start_cap - goal_time


#combine all above arrays into 1 big array and save this in a csv file 
big_array = np.column_stack([trial_count, block_count, frame_count, start_cap, goal_time, compare_time, end_cap, 
                              action_started, action_ended])
big_array = np.column_stack([big_array, store_pics, store_colors, store_accuracies, store_corResps, store_RTs])
big_DF = pd.DataFrame.from_records(big_array)
big_DF.columns = np.concatenate([['Trial_number', 'block_count', 'Frame_count', 'Reading_started', 'Wanted_start', 'Deviation_wanted', 
                  'Reading_ended', 'Action_started', 'Action_ended'], OASIS_columns, ['Affect', 'color', 'accuracy', 'corResp', 'RT']])

if try_out == 0: 
    output_file = os.path.join(video_directory, str("output_participant" + str(number) + '.csv'))
else: 
    output_file = 'Stored_info_test.csv'
big_DF.to_csv(output_file, index = False)


meta_file = os.path.join(video_directory, str("demographics_participant" + str(number) + '.csv'))
info.pop('Naam')
meta_data_df = pd.DataFrame(info, index = [0])
meta_data_df.to_csv(meta_file, index = False)
