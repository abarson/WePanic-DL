"""
basics.py is a library of utility functions meant
to expedite boiler plate actions like loading up data
"""

import sys
import os
import csv

def check_exists_create_if_not(directory, suppress=False):
    """
    check whether a directory exists -- create it if it doesn't
    
    args:
        directory: name of directory to test

    returns:
        Nothing
    """
    
    if "." in directory:
        raise ValueError("looks like %s isn't a valid directory name" % directory)

    if not os.path.isdir(directory):
        if not suppress:
            print("[check_exists_create_if_not] making a dir: %s" % directory)
        os.makedirs(directory)

    return directory 

def fetch_path(subj, data_dir):

    subject = "S" + ("0" * (4-len(subj))) + subj
    full_path = os.path.join(data_dir, subject)
    return full_path

def progressBar(value, endvalue, bar_length=20):
    """
    Stolen from StackOverflow. For that dank looking progress bar.
    """
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r[{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

#####################################################
###################################           ####### 
################################   DEPRECATED    ####
###################################           #######
#####################################################

class CSV_Helper():
    """
    Use this for looking up data in DeepLearningClassData.csv,
    and writing to some specified csv output.
    """
    def __init__(self, look_up_csv, write_to_csv, header=None):
        self.look_up_csv = look_up_csv
        self.write_to_csv = write_to_csv 
        self.csv_writer = csv.writer(write_to_csv, delimiter=',',
                                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if header is None:
            self.create_header() 
        else:
            self.generate_header(header)

    def create_header(self):
        first_line = self.look_up_csv.readline()
        first_line = first_line.rstrip().split(",")
        header = [first_line[0], "PARTITION"]
        header.extend(first_line[1:])
        self.csv_writer.writerow(header)
    
    def generate_header(self, header):
        self.csv_writer.writerow(header)

    def look_up(self, subj_name):    
        size = len(subj_name)
        for line in self.look_up_csv:
            if line[0: size] == subj_name:
                self.look_up_csv.seek(0)
                return line[size+1:].rstrip().split(",")
        self.look_up_csv.seek(0)
        return "?"
    
    def write_to(self, subj_name, partition, specs, num_partitions):
        params = [subj_name, partition]
        if num_partitions[0] <= partition:
            specs[0] = ""
            specs[1] = ""
        if num_partitions[1] <= partition:
            specs[2] = ""
            specs[3] = ""
        params.extend(specs)
        self.csv_writer.writerow(params)
        
    def release(self):
        self.look_up_csv.close()
        self.write_to_csv.close()
        
def csv2data(filename, has_header=True):
    """
    read the data in a .csv file with a header
    and return a matrix M of data.

    args:
        --> filename : the .csv file
        --> has_header : defaulted to True; specify whether
                         the input file has a header or not
    returns:
        --> data matrix M: the rows x columns of the .csv,
                         the header of the file
    """
    if os.path.exists(filename):

        if filename.endswith(".csv"):
            M = []
            with open(filename, "r") as csvfile:
                reader = csv.reader(csvfile)

                header = None

                if has_header:
                    header = next(reader, None)

                for row in reader:
                    M.append(row)

                return M, header

        else:
            raise ValueError("%s doesn't end in .csv!" % filename)

    else:
        raise FileNotFoundError("Could not locate %s" % filename)


