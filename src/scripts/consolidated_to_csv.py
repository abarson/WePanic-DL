"""
Convert the input csv to one that mirrors the dir/subdir
format in consolidated data directory
"""

import sys, os
import we_panic_utils.basic_utils.basics as base

def usage():
    print("[usage]: python %s <partition-csv> <consolidated_csv>" % sys.argv[0])
    sys.exit()
 
def parse_input():
    try:
        dir_ = sys.argv[1]
        csv_out = sys.argv[2]

        if os.path.exists(dir_):
            return dir_, csv_out

        else:
            raise OSError("%s not found" % dir_)

    except IndexError:
        usage()

if __name__ == "__main__":
    csv_, csv_out = parse_input()
    

    mat, header = base.csv2data(csv_)
    csv_ = open(csv_, "r")
    output_csv = open(csv_out,"w")
    
    header = ["SAMPLE","HEART RATE","RESPIRATORY RATE","HEART RATE CLASS"] 
    csvh = base.CSV_Helper(csv_, output_csv, header=header)
    #header = ["SUBJECT","TRIAL","PARTITION","HEART RATE","RESPIRATORY RATE","HEART RATE CLASS"]
    for row in mat:
        subj, tr, part = [int(r) for r in row[:3]]
        fmt_slug = "S%04d_t%d_p%d" % (subj, tr, part)
        
        newr = [fmt_slug]
        newr.extend(row[3:])

        csvh.csv_writer.writerow(newr)
    
    csvh.release()

    print("done!")
