"""
video_core.py is a subsection of basic_utils library that is meant 
for video processing. 
"""
from PIL import Image
import sys
import os
import cv2
from we_panic_utils.basic_utils.basics import check_exists_create_if_not, fetch_path, progressBar 

import subprocess
FPS = 30

BASH_COMMAND = 'ffmpeg -i {} -q:v 1 -vf setpts={}*PTS {}' 
GET_DURATION_COMMAND = "ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 {}"
SPLIT_COMMAND = "ffmpeg -i {} -vcodec copy -acodec copy -ss {} -t {} {}"

HALVE_COMMAND = "ffmpeg -i {} -vcodec copy -acodec copy \
    -ss {} -t {} {}"


def video_file_exists(filename):
    """
    Return whether or not this file exists and is a
    video file (i.e, ends in [.MOV | .mov]

    args:
        --> filename : the file to test

    returns:
        --> boolean of whether this file is a video file
        --> error message if something bad occurred
        --> the filename minus the extension and parent dirs
    """
    does_exist = os.path.exists(filename)
    is_vid = filename.endswith(".mov") or filename.endswith(".MOV")

    if does_exist and is_vid:
        if len(filename.split("/")) > 1:
            parent = filename.split("/")[-2]
        
        else:
            parent = ""

        no_ext = filename.split("/")[-1][:-4]
        
        # example : S0001/Trial1
        return_me = os.path.join(parent, no_ext)
        return True, None, return_me

    output_string = []
    if not is_vid:
        output_string.append("%s isn't a .MOV or .mov" % filename)

    if not does_exist:
        output_string.append("%s not found" % filename)

    error = " AND ".join(output_string)
    return False, error, None


def video_file_to_frames(filename, resize=None, output_dir=None, suppress=False, clip=2):
    """
    Convert a video file to individual frames
    
    Added by Adam:
    This method will also ensure that all videos are reduced to 30fps. For a
    60fps video, for example, every other frame will be skipped.
    

    Added by dan: 6/8/18
    video_file_to_frames now proudly supports frame resizing -- just pass in a tuple
    of the form (xdim, ydim) 
    args:
        --> filename : video file to convert
        --> output_dir (optional): the desired output directory
                               for the frames

        --> suppress : boolean to suppress messages or not
    returns:
        --> list of image filenames
    some facts:
    ----------
        1) This procedure will save the png images in an output directory
           called '$(filename_dir)/$(filename_frames)' if no directory is specified.

        2) if an output directory IS specified but not found, this procedure
           will create the output directory, with a subdir called %filename_frames

        3) Stack overflow source:
        [https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames]
    """
    # check the video's existence
    vid_valid, err, no_ext = video_file_exists(filename)
    #print("vid : {}".format(vid_valid))
    
    if resize:
        print(resize)
        resize = tuple(resize)
        assert len(resize) == 2, 'need xdim and ydim'

    if vid_valid:
        if output_dir:
            output_dir = os.path.join(output_dir, "%s_frames" % no_ext)
            check_exists_create_if_not(output_dir, suppress=suppress)

        else:
            output_dir = "%s_frames" % no_ext
            check_exists_create_if_not(output_dir, suppress=suppress)

        # have output directory, now need to create the framesies
        vidcap = cv2.VideoCapture(filename)
        FPS = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
        total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        success, image = vidcap.read()
        
        image_names = []
        count = 0
        success = True

        if not suppress:
            print("[video_file_to_frames]-- extracting frames from %d fps video with %d frames" % (FPS, total))

        # while there is a next image
        while success:
            
            # frame resizing
            if resize:
                image = cv2.resize(image, resize)

            pth = ""
            if count >= FPS * clip and count < total - (FPS * clip):
                if FPS == 60:
                    if count % 2 == 0:        
                        pth = os.path.join(output_dir, "frame-%05d.png" % ((count-FPS*clip)/2))
                        image_names.append(pth)
                        cv2.imwrite(pth, image)
                else:
                    pth = os.path.join(output_dir, "frame-%05d.png" % (count-FPS*clip))
                    image_names.append(pth)
                    cv2.imwrite(pth, image)

             
            success, image = vidcap.read()
             
            if not suppress and pth != "":
                if success:
                    sys.stdout.write("\r[video_file_to_frames]-- writing [%s]" % pth)
                    sys.stdout.flush()
                
                else:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

            count += 1
          
        if not suppress:
            print("\n[video_file_to_frames]-- clipped [%d] seconds off of each end of video" % clip)
        #clip off two seconds of the video
        
        #i = 0
        #size = len(image_names)
        #for video_frame in os.listdir(output_dir): 
        #    if i < clip * FPS:
        #        pth = os.path.join(output_dir, video_frame)
        #        os.remove(pth)
        #    elif i > size - (clip * FPS):
        #        pth = os.path.join(output_dir, video_frame)
        #        os.remove(pth)
        #    elif FPS == 60 and i % 2 == 1:
        #        pth = os.path.join(output_dir, video_frame)
        #        os.remove(pth)
        #    i+=1
        return image_names
    
    # a problem occurred
    else:
        raise ValueError(err)


def video_dir_to_frame_dir(video_dir, output_dir, suppress=False):
    """
    create a directory that contains subdirectories
    that contain all of the frames of the movies contained
    within the video_dir

    args:
        video_dir : directory with videos
        output_dir : location to place the output frames
        suppress (optional) : display output 
    returns:
        imgs_captured : a list of image framenames from all 
                        videos in video dir
    """
    
    # path exists?
    if os.path.exists(video_dir):
        
        # path is a dir?
        if os.path.isdir(video_dir):
            contents = os.listdir(video_dir)
            movies = [os.path.join(video_dir, cont) for cont in contents if cont.lower().endswith(".mov")]

            # path contains .MOV or .mov files ??
            if len(movies) > 0:
                
                imgs_captured = []

                check_exists_create_if_not(output_dir, suppress=suppress)

                for mov in movies:
                    framenames = video_file_to_frames(mov, output_dir=output_dir, suppress=suppress)
                    imgs_captured.extend(framenames)

                return imgs_captured

            else:
                raise FileNotFoundError("%s contains no video files" % video_dir)

        else:
            raise ValueError("%s isn't a directory" % video_dir)

    else:
        raise FileNotFoundError("%s not found" % video_dir)


    
def move_frames(source_dir, partitioned_frames, output_dir):
    """
    Helper method to move a selection of frames from the source directory
    into a new output partition directory.

    args:
        source_dir : the origin directory
        partitioned_frames : a list of frames that are being moved out of source directory
        output_dir : the directory where selected frames are being placed
    """
    check_exists_create_if_not(output_dir,suppress=True)
    current_index = 0
    for frame in partitioned_frames:
        frame_path = os.path.join(source_dir, frame)
        source_path = os.path.join(output_dir, "frame" + str(current_index) + ".png")
        os.rename(frame_path, source_path)
        current_index+=1

def resize_frame_dir(frame_dir, output_dir, width=224, height=224):
    """
    Copy and resize frames in given directory.
    """
    if not os.path.exists(frame_dir):
        raise FileNotFoundError("Error: path {} does not exists".format(frame_dir))
    if not os.path.isdir(frame_dir):
        raise IOError("Error: path {} is not a directory".format(frame_dir))
    
    check_exists_create_if_not(output_dir, suppress=True)

    print("[resize_frame_dir]: RESIZING {} -> {}".format(frame_dir, output_dir))
    listed_directory = os.listdir(frame_dir)
    num_partitions = len(listed_directory)
    completed_partitions = 0

    for frame in os.listdir(frame_dir):
        current_frame_dir = os.path.join(frame_dir, frame)
        img = Image.open(current_frame_dir)
        img = img.resize((width, height), Image.ANTIALIAS)
        output_path = os.path.join(output_dir, frame)
        img.save(output_path) 
        completed_partitions += 1
        progressBar(completed_partitions, num_partitions)
    print()


def change_speed(video_path, new_name, factor):
    """
    Change the speed of a video by some provided factor. The resulting video created
    will be saved to a path specified by the user.

    This method uses the FFMPEG video library to assist in changing the speed of a video.
    For more information, go to https://www.ffmpeg.org/. To assist in ease of use,
    the FFMPEG commands used by this method are declared as constants in the same module.

    params:
        video_path : the complete path of the video to be augmented
        new_name : the desired name of the resulting video
        factor : the factor by which to change the speed of the video
    """

    command = BASH_COMMAND.format(video_path, factor, new_name)
    print("[{}]: CHANGING SPEED of video {} by a factor of {}".format(sys.argv[0], video_path, str(factor)))
    p = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE)
    p.wait()
    p.kill()
    __clip_video(new_name, factor)

def __clip_video(video_path, factor):
    """
    Utility method to clip a video, called by the public change_speed method.
    This method will ensure that any videos that have been augmented will still
    be under 30 seconds. For videos that have been slowed down, for example,
    30 seconds will be cropped out of the middle of the video.

    params:
        video_path : the path of the video to be clipped
        factor : the factor by which the speed of the video was changed
    """
    
    command = GET_DURATION_COMMAND.format(video_path)
    p = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
    print(out)
    out = int(float(out))
    print(out)
    if factor < 1:
        end_time = "00:00:" + str(int(out*factor))
        command = HALVE_COMMAND.format(video_path, "00:00:00", end_time, 'temp.mov') 
        p = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        p.wait()
        p.kill()
        os.remove(video_path)
        os.rename('temp.mov', video_path)
    
    if out > 30:
        dif = int((float(out)-30)/2)
        start = str(dif)
        if len(start) == 1:
            start = "0" + start
        start = "00:00:" + start
        command = HALVE_COMMAND.format(video_path, start, "00:00:30", 'temp.mov') 
        p = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        p.wait()
        p.kill()
        os.remove(video_path)
        os.rename('temp.mov', video_path)

#####################################################
###################################           ####### 
################################   DEPRECATED    ####
###################################           #######
#####################################################
def partition_frame_dir(frame_dir, output_dir, num_seconds=2, front_trim=60, end_trim=60, 
                        capacity_tolerance=1.0):
    """
    Given a directory of frames, this method partitions them into several subdirectories,
    such that each directory contains num_seconds*FPS frames.

    args:
        frame_dir : directory containing frames
        output_dir : directory to place partitions
        num_seconds (optional) : the number of seconds that make up each partition
        front_trim (optional) : the number of frames to ignore from beginning of directory
        end_trim (optional) : the number of frames to ignore from end of directory
        capacity_tolerance (optional) : how full an acceptable partition must be 
    
    return:
        the number of partitions created
    """
    
    if not os.path.exists(frame_dir):
        raise FileNotFoundError("provided directory |%s| not found" % frame_dir)
    if not os.path.isdir(frame_dir):
        raise IOError("provided path |%s| is not a directory" % frame_dir)
    if front_trim < 0 or end_trim < 0 or num_seconds < 0:
        raise ValueError("num_seconds, front_trim and end_trim must be positive")
    
    listed_directory = os.listdir(frame_dir)
    num_frames = len(listed_directory)

    #rough estimation to determine frame rate
    is_60_fps = num_frames > 1600
    
    print('[partition_frame_dir]: PARTITIONING {} -> {}'.format(frame_dir, output_dir))
    print("[partition_frame_dir]: Found {} frames".format(num_frames), ("(60fps)" if is_60_fps else "(30fps)"))
    iteration = 0
    num_partitions = 0
    current_partition = []
    
    #The next five lines are only used for the progress bar output. Ignore it if you want.
    eligible_frames = (num_frames - front_trim - end_trim + 1) // (2 if is_60_fps else 1)
    total_partitions = eligible_frames // (num_seconds * FPS)
    left_over = eligible_frames - total_partitions * num_seconds * FPS
    if left_over / (num_seconds * FPS) >= capacity_tolerance:
        total_partitions += 1

    while iteration < num_frames - end_trim:
        if iteration >= front_trim:
            if not is_60_fps:
                current_partition.append(listed_directory[iteration])
            elif iteration % 2 == 0:
                current_partition.append(listed_directory[iteration])
            
            if len(current_partition) >= num_seconds*FPS:
                next_output_dir = os.path.join(output_dir, str(num_partitions))
                move_frames(frame_dir, current_partition, next_output_dir)           
                num_partitions+=1
                current_partition = []
                progressBar(num_partitions, total_partitions)
        iteration+=1

    #Handle any leftover frame lists that did not reach full capacity due to trimming.
    #If the frame list is acceptably full, then include it with the other partitioned directories
    if current_partition:
        if len(current_partition) / (num_seconds*FPS) >= capacity_tolerance:
            next_output_dir = os.path.join(output_dir, str(num_partitions))
            move_frames(frame_dir, current_partition, next_output_dir)
            num_partitions+=1
            progressBar(num_partitions, total_partitions)

    print()
    return num_partitions


def handle(video_path, factor):
    command = GET_DURATION_COMMAND.format(video_path)
    p = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
   
    if factor == 2:  #if the video was slowed down, split it into two 30 second parts and throw out the second part
        split_name = video_path.split('.')[0]
        end_time_1 = int(float(out) / 2)
        end_time_2 = int(float(out))
        if len(str(end_time_1)) < 2:
            end_time_1 = int("0" + str(end_time_1))
        if len(str(end_time_2)) < 2:
            end_time_2 = int("0" + str(end_time_2))
        end_time_1 = "00:00:" + str(end_time_1)
        if end_time_2 > 60:
            end_time_2 = "00:01:" + '0' + str(end_time_2 - 60)
        else:
            end_time_2 = "00:00:" + str(end_time_2)
        command = SPLIT_COMMAND.format(video_path, 0, end_time_1, split_name + '_first_half.mov') 
        p = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        p.wait()
        p.kill()
        os.remove(video_path)
        os.rename(split_name + '_first_half.mov', video_path)

    else:  #if the video was sped up, cut it in half and reverse it
        halved_name = video_path.split('.')[0] + '_halved.mov'
        end_time = "00:00:" + str(int(float(out)/2))
        command = HALVE_COMMAND.format(video_path, "00:00:00", end_time, halved_name) 
        p = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        p.wait()
        p.kill()
        looped_name = video_path.split('.')[0] + '_looped.mov'
        command = ['ffmpeg','-i',halved_name,'-filter_complex','[0:v]reverse,fifo[r];[0:v][0:a][r][0:a]concat=n=2:v=1:a=1 [v] [a]', '-map', '[v]', '-map', '[a]', looped_name] 
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        p.wait()
        p.kill()
        os.remove(halved_name)
        os.remove(video_path)
        os.rename(video_path.split('.')[0] + '_looped.mov', video_path)


