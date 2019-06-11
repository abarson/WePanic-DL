import numpy as np
import process_mov as pm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def arguments():
    parser = ArgumentParser(description='condense a sequence')
    parser.add_argument('movs',
                        help='list of .mov files',
                        nargs='+',
                        type=pm.mov)

    #parser.add_argument('--output-file','-o',
    #                    dest='output',
    #                    help='path to output file (default to stdout)',
    #                    type=pm.csv,
    #                    default='default')
    
    return parser

def spatial_avg(frame):
    return np.mean(np.ravel(frame))

if __name__ == '__main__':
    args = arguments().parse_args() 
    for mov in args.movs:
        gen = pm.frame_generator(mov, 
                                nosubsamp=True,
                                preprocs=[pm.resize, pm.normalize, spatial_avg])
       
        ts = list(gen)
        plt.plot(ts, marker='o', label=mov)
        plt.xlabel('Frame #')
        plt.ylabel('Spatial average')
    plt.legend()
    plt.show()
