"""
Catalog partitions in a .csv of the form:

SUBJECT     TRIAL       PARTITION       HEART RATE      RESPIRATORY RATE
"""

import re
import os
import sys
import we_panic_utils.basic_utils.basics as base

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def usage():
    print("[usage]: python %s <partition-dir> <master_csv>" % sys.argv[0])
    sys.exit();

def parse_input():
    try:
        dir_ = sys.argv[1]
        csv_ = sys.argv[2]

        if os.path.isdir(dir_) and os.path.exists(csv_):
            return dir_, csv_

        else:
            raise OSError("%s not found" % dir_)

    except IndexError:
        usage()

if __name__ == "__main__":
    dir_, csv_ = parse_input()
    hrhigh = 0
    hrlow = 0
    mat, header = base.csv2data(csv_)
    
    csv_ = open(csv_, "r")
    #print(header)
    output_csv = open("partitions_cons.csv","w")
    
    exclude = [".DS_Store","._.DS_Store"]
    
    header = ["SUBJECT","TRIAL","PARTITION","HEART RATE","RESPIRATORY RATE","HEART RATE CLASS"] 
    csvh = base.CSV_Helper(csv_, output_csv, header=header)
    #csvh.generate_header(header)
    #print(mat)
    # gives subjects
    for child in sorted(os.listdir(dir_)):
        if os.path.isdir(os.path.join(dir_, child)) and child not in exclude:
            pth = os.path.join(dir_, child)
            
            # gives TrialN_frames
            for grandchild in sorted(os.listdir(pth), key=numericalSort):
                if grandchild not in exclude:
                    
                    fullpth = os.path.join(pth, grandchild)
                    
                    # gives partition namees
                    for greatgc in sorted(os.listdir(fullpth), key=numericalSort):
                        if greatgc not in exclude:
                            
                            # write to output_csv the follwoing
                            # SUBJECT
                            # TRIAL
                            # PARTITION
                            # HEART RATE 
                            # RESPIRATORY RATE

                            SUBJECT = str(int(child[1:]))
                            TRIAL   = grandchild.split("_")[0][-1]
                            PARTITION = str(int(greatgc))
                            row = mat[int(SUBJECT)-1]
                            
                            HEART_RATE, RESPIRATORY_RATE = "", ""
                            HEART_RATE_CLASS = "LOW"
                            if int(TRIAL) == 1:
                                HEART_RATE, RESPIRATORY_RATE = row[1:3]
                            else:
                                HEART_RATE, RESPIRATORY_RATE = row[3:]

                            if int(HEART_RATE) >= 100:
                                HEART_RATE_CLASS = "HIGH"
                                hrhigh+=1

                            else:
                                hrlow+=1

                            csvh.csv_writer.writerow([SUBJECT, TRIAL, PARTITION, HEART_RATE, RESPIRATORY_RATE,HEART_RATE_CLASS])
                            #print([SUBJECT, TRIAL, PARTITION])
    csvh.release()

    print("done! %d samples with high heart rate, %d samples with low heart rate" % (hrhigh,hrlow))
