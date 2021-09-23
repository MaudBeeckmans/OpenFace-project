### script for converting videos with OpenFace - Luc Vermeylen (Feb 2021)
# note: requires OpenFace folder to be installed (tested with OpenFace_2.2.0_win_x64)
# to do: generalize to multiple subjects

#1. Preparations: load the correct libraries, define the correct working directory & create some useful variables

# Libraries
library(here) # easy function to refer to current path, also when not in a Rproject
library(tidyverse)
library("rstudioapi")


#Goal of the 2 lines of code below: make sure the working directory is "Final_versions/OpenFace_Procesisng"
  #In this folder the final output file will be uploaded 
source_dir = dirname(getActiveDocumentContext()$path)
setwd(source_dir) #set wd to the directory of your source file 

processed <- TRUE

n_trials_pp <- c()#store amount of trials for each pp
meta_data <- c()#this will contain the meta-data of all files eventually 
# which participants do you want to include in the output file? 
  #(each participant number should be filled in manually)
pp_numbers <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)  #change to number of pp we've recorded 
# How many blocks (n_blocks) did you record per participant and how are these nubered (blocks)? 
  #(Each block is a separate folder, each participant has a separate folder for each block)
blocks <- c(1, 2, 3)
n_blocks <- 3

# Set paths
  #Below: where is your FeatureExtraction.exe file (from OpenFace software)
OpenFace_path <- paste0("C:\\Users\\Maud\\Documents\\Psychologie\\1ste_master_psychologie\\Masterproef\\OpenFace_2.2.0_win_x64", '/', "FeatureExtraction.exe")

#2. The big loop: loop over all participants to 
  #(1) Process all video's 
  #(2) create meta_data containing all extra information from the experiments for all participants

#PP_loop: loops over all participants that you've defined at line 20 'pp_numbers'
for (this_pp in pp_numbers){
#Where can we find all the information from the experiment for this participant? 
pp_path <- paste0("C:\\Users\\Maud\\Documents\\Psychologie\\1ste_master_psychologie\\Masterproef\\Participanten_Pilot", 
                    "\\participant", this_pp)
for (this_block in blocks){
#Where can we find the video's for this block and this participant ?   
video_files_path <- paste0(pp_path, "\\block", this_block)
#Where do I want to store all the information received after processing a video? 
  # an output folder will be created if it does not exist
output_files_path <- paste0(getwd(),'/',"processed_videos", "/participant", this_pp, "/block", this_block ) 
video_type <- ".avi"  # for selecting the relevant file types

# Get a list of all the video names (ordered_files) and the video's that have already been processed previously (Processed_videos)
  #It is important that the ordered_files is ordered based on trial number, the processed_files does not have to be ordered on trial number
# Processed_files: a list containing the names of which video's have already been processed with OpenFace before running this script
  #Will be used to check whether a video should be sent through OpenFace still 
processed_files <- dir(path = output_files_path, pattern = ".csv", full.names = F)
# Files: a list containing the names of all the video's from the experiment for this pp. and this block 
  #Will be used to loop over (process all 'not yet processed' video's with OpenFace)
files <- dir(path = video_files_path, pattern = video_type, full.names = T)
  #this again is ordered alphabetically, but we want it to be ordered based on trial number!
  #Therefore: the 4 lines below!
ordered_files <- c()
for (this_trial in 0:(length(files)-1)){
  search <- paste0("pp", this_pp, "_block", this_block, "_trial", this_trial, "_")
  index <- grep(search, files)
  ordered_files <- rbind(ordered_files, files[index])
}

# Process the videos with OpenFace using the command line ("shell") (see here for all arguments: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments)

for (i in 1:length(ordered_files)) {
  #check_processed: this is the argument that has to be looked for in the 'processed_files' list 
    #to check whether a video was already processed before
  check_processed = paste0('pp', this_pp,  '_block', this_block, '_trial', (i-1), '_')
  #already_processed: contains the output of whether the current file (check_processed) was already processed
    #grep-argument: looks for a match, output: list containing the locations of where the match(es) were found
    # thus length of this list will be 0 when no matches were found and the file should still be processed 
  already_processed = grep(check_processed, processed_files)
  if (length(already_processed) != 0){
    #only let shell run when video hasn't been processed yet 
      #(when .csv file doesn't exist yet in output_folder)
    #print(paste0("file",  ordered_files[i], "was processed already"))
    #print(i)
  }else{
  # print some stuff to follow in the console where we are
  print("---------------------------------------------------------------")
  print(paste0('----------------------------- ',i,' -------------------------'))
  print("---------------------------------------------------------------")
  # the actual shell command
    #Important here: whether you'd like to use AU_static or AU_dynamic!
  shell(paste0(OpenFace_path, # the path to open face
               ' -f ', ordered_files[i], # the file to process
               ' -au_static ',' -out_dir ', output_files_path)) # the output folder
  print(ordered_files[i])
  }
}
}
#this_meta_data: contains the extra information from the experiment (from csv file) for this participant 
this_meta_data <- read.csv(paste0(pp_path,"/output_participant", this_pp, ".csv"))
#Add a new column named 'pp_number' 
this_meta_data$pp_number <- rep(this_pp, each = length(this_meta_data$Trial_number))  #add participant number to the meta-data file
#meta_data: contains the extra information from the experiment for all participants 
meta_data <- rbind(meta_data, this_meta_data)
n_trials <- length(ordered_files)
n_trials_pp <- rbind(n_trials_pp, n_trials )

}

#3. Create the data matrix containing all information 
  #(OpenFace_output & experimental_output for each pp and each trial)

# 3(A) Concatenate all csv files obtained from processing the video's through OpenFace
#First: create a list containing the files of all the processed_videos in the correct order (based on pp_number)
  #ordered_output: is created to afterwards correctly concatenate all .csv output_files from OpenFace processing 
  #ordered_fnames: is created to afterwards add the video_name to the big data file 
ordered_output <- c()
ordered_fnames <- c()
for (this_pp in pp_numbers){
  for (this_block in blocks){
    output_files_path <- paste0(getwd(),'/',"processed_videos", "/participant", this_pp, "/block", this_block)
    #when list.files: the files are stored in alphabetic order, which is NOT the order we want
    output_files <- list.files(path=output_files_path, pattern = "*.csv", full.names=T)
    out_fnames <-list.files(path=output_files_path, pattern = "*.csv", full.names=F)
    amount_trials <- 150
    for (this_trial in 0:(amount_trials-1)){
      search <- paste0("pp", this_pp, "_block", this_block, "_trial", this_trial, "_")
      index <- grep(search, output_files)
      ordered_output <- rbind(ordered_output, output_files[index])
      ordered_fnames <- rbind(ordered_fnames, out_fnames[index])
      }
  }
}

#Second: bind row-wise all the different OpenFace_created_output_files 
#for all the elements in the ordered_output, read the csv and store this in data, I think now the order is correct 
data <- do.call(rbind,lapply(ordered_output,read.csv))
#data <- do.call(rbind,lapply(ordered_output,print))  #check that the order is correct now

# 3(B) Add metadata from the experiment to each processed video's .csv file (for participant, trial and condition info)
  #all_data contains both the meta_data and the openface_data for all participants
all_data <- cbind(meta_data,data)
  #Add the video_name to the big data matrix (all_data) for clarity 
all_data$video <- rep(ordered_fnames, each = length(unique(data$frame)))

#4. Save the big data matrix as an RDS file 

RDS_name <- paste0("data_processed_concat", pp_numbers[1], "to", tail(pp_numbers, 1),'_AUstatic', ".rds")
saveRDS(all_data, file =  RDS_name)# .rds is much smaller and faster than .csv when files get really big

