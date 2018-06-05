"""
Consolidate the data to one directory of subdirs
where the subdir name reflects the subject_trial_partition
"""

import os
import sys
import shutil

def usage():
    print("[usage] %s <directory> <consolidated_out_dir>" % sys.argv[0])
    sys.exit()

def parse_input(): 
    try:
        dir_ = sys.argv[1]
        out_dir_ = sys.argv[2]
        if os.path.isdir(dir_):
            return dir_, out_dir_

        else:
            raise OSError("%s not found" % dir_)

    except IndexError:
        usage()

if __name__ == '__main__':
    #pass
    dir_, out_dir_ = parse_input()
    os.makedirs(out_dir_)
    exclude = [".DS_Store","._.DS_Store"]
     
    for child in sorted(os.listdir(dir_)):
        if os.path.isdir(os.path.join(dir_, child)) and child not in exclude:
            pth = os.path.join(dir_, child)
            
            # gives TrialN_frames
            for grandchild in sorted(os.listdir(pth)):
                if grandchild not in exclude:
                    
                    fullpth = os.path.join(pth, grandchild)
                    
                    # gives partition namees
                    for greatgc in sorted(os.listdir(fullpth)):
                        if greatgc not in exclude:
                 
                            SUBJECT = child
                            TRIAL   = grandchild.split("_")[0][-1]
                            PARTITION = greatgc

                            slug = "%s_t%s_p%s" % (SUBJECT, TRIAL, PARTITION)
                            
                            subdir = os.path.join(out_dir_,slug)
                         
                            os.makedirs(subdir)
                            
                            partpth = os.path.join(fullpth,greatgc)
                            contents = os.listdir(partpth)
                        
                            for f in contents:
                                cpath = os.path.join(partpth,f)
                                
                                sys.stdout.write("\r" + cpath)
                                sys.stdout.flush()

                                shutil.copy2(cpath,subdir)


    print("done!")
