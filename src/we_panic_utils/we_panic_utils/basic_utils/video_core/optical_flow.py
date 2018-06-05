import cv2
import numpy as np
import pickle
from PIL import Image
import os
import glob
import gc
import sys
from ..basics import check_exists_create_if_not 

def optical_flow_of_first_and_rest(frames):
    count = 0
    frame1 = Image.open(frames[0])
    
    #only compare the optical flow of first image to every other image
    prvs = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2GRAY)
    
    all_hor = []
    all_ver = []
    for pth in frames[1:]:
        next_ = cv2.cvtColor(np.array(Image.open(pth)), cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 0.6, 3, 10, 3, 5, 1.2, 0)
        horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        all_hor.append(horz)
        all_ver.append(vert)
   
    cv2.destroyAllWindows()
    return all_hor, all_ver

def write_optical_flow(path, width):
    count = 0

    try:
        all_horz, all_vert = [], []
        frame_paths = sorted(glob.glob(path + "/*"))

        frame1 = Image.open(frame_paths[0])

        prvs = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2GRAY)
        
        check_exists_create_if_not(os.path.join(path, "flow_v"))
        check_exists_create_if_not(os.path.join(path, "flow_h"))
        
        flow_v = os.path.join(path, "flow_v")
        flow_h = os.path.join(path, "flow_h")

        for pth in frame_paths[1:]:
            next_ = cv2.cvtColor(np.array(Image.open(pth)), cv2.COLOR_RGB2GRAY)
            fname = pth.split('/')[-1]
            count += 1

            if count % width == 0:
                sys.stdout.write("[\r%s : %04d]" % (pth, count))
                sys.stdout.flush()
                
                flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 0.6, 3, 10, 3, 5, 1.2, 0)
                horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
                vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)

                horz = horz.astype('uint8')
                vert = vert.astype('uint8')
                
                cv2.imwrite(os.path.join(flow_h, fname), horz)
                cv2.imwrite(os.path.join(flow_v, fname), vert)
                
                prvs = next_
                
        cv2.destroyAllWindows()

    except Exception as e:
        print(e)
        return count
