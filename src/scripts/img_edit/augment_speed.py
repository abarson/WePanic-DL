import csv
import subprocess
import os
import sys
import pandas as pd

import we_panic_utils.basic_utils.basics as base
import we_panic_utils.basic_utils.video_core as vc

UPPER_THRESHOLD = 230
LOWER_THRESHOLD = 35

if __name__ == "__main__":
    try:
        data_dir = sys.argv[1]
        master_csv = sys.argv[2]
        selected = sys.argv[3]
    except:
        print("\tUsage: <data_dir> <master_csv> <selected_subjects.csv>")
    else:       
        #gives a list of values between 0.5 and 2 without 1
        speed_changes = [round(x*0.1, 1) for x in range(5, 21) if x != 10]

        selected_df = pd.read_csv(selected)
        data = base.csv2data(master_csv)[0]

        filtered = sorted(zip(list(selected_df["Subject"]), list(selected_df["Trial"])), key=lambda x: x[1])
         
        with open("aug_log.txt", "w") as aug_log, open(master_csv, 'a') as master:     
            master_writer = csv.writer(master)
            master_writer.writerow([])
            for subj, trial in filtered:
                subj_row = data[subj-1]
                heart_rate, resp_rate = 0, 0
                if trial == 1:
                    heart_rate, resp_rate = float(subj_row[1]), float(subj_row[2])
                elif trial == 2:
                    heart_rate, resp_rate = float(subj_row[3]), float(subj_row[4])
               
                subj_origin = os.path.join(data_dir, "S%04d" % subj, "Trial%d.MOV" % trial)
                assert os.path.exists(subj_origin)
                
                new_subj = "%d%02d" % (trial, subj)
                new_subj_path = "S0%s" % new_subj

                base.check_exists_create_if_not(os.path.join(data_dir, new_subj_path))
                
                subj_target_fast = os.path.join(data_dir, new_subj_path, "Trial1.MOV")
                subj_target_slow = os.path.join(data_dir, new_subj_path, "Trial2.MOV")
                
                """     s   hr1  rr1  hr2  rr2   """
                """     |    |    |    |    |    """ 
                row = ["",  "",  "",  "",  ""]
                
                row[0] = new_subj

                if heart_rate * 2 <= UPPER_THRESHOLD:
                    row[1] = heart_rate * 2
                    row[2] = (heart_rate * 2) / 4
                    selected_df.loc[len(selected_df)] = [new_subj, 1]
                    vc.change_speed(subj_origin, subj_target_fast, 0.5)
                else:
                    log_str = "Not doubling the speed of subject {} trial {}: resulting" \
                            " heart rate would have been {}".format(subj, trial, heart_rate * 2)
                    print(log_str)
                    aug_log.write(log_str + '\n')

                if heart_rate / 2 >= LOWER_THRESHOLD:
                    row[3] = heart_rate / 2
                    row[4] = (heart_rate / 2) / 4 
                    selected_df.loc[len(selected_df)] = [new_subj, 2]
                    vc.change_speed(subj_origin, subj_target_slow, 2)
                else:
                    log_str = "Not halving the speed of subject {} trial {}: resulting" \
                            " heart rate would have been {}".format(subj, trial, heart_rate / 2)
                    print(log_str)
                    aug_log.write(log_str + '\n')
                
                master_writer.writerow(row) 
            selected_df.to_csv(selected, encoding='utf-8', index=False)
