import sys
import os
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm 

LOSS = 1
VAL_LOSS = 3

def compare_losses(directories):
    log_collections = []
    for d in directories:
        logs = [os.path.join(d, log) for log in os.listdir(d) if '.log' in log]
        logs.sort()
        log_collections.append(logs)
    
    avg_trains, avg_vals = [], []
    for log_collection in log_collections:
        avg_train, avg_val = __average_train_val_losses(log_collection)
        avg_trains.append(avg_train)
        avg_vals.append(avg_val)

    plt.figure('Average Cross Validation Training Loss and Validation Loss')
    color=cm.rainbow(np.linspace(0,1,len(directories)))
    patches = [mpatches.Patch(color=color, label=model_name.split('/')[-1]) for model_name, color in zip(directories, color)]

    plt.subplot(211)
    plt.title('Average Cross Validation Training Loss and Validation Loss')
    plt.ylabel('Average Training Loss')
    for i, (loss, c) in enumerate(zip(avg_trains, color)):
        plt.plot([j for j in range(len(loss))], loss, c=c, label='Fold {}'.format(i))

    plt.legend(handles=patches)

    plt.subplot(212)
    plt.ylabel('Average Validation Loss')
    for val_loss, c in zip(avg_vals, color):
        plt.plot([i for i in range(len(val_loss))], val_loss, c=c) 

    plt.tight_layout()
    
    #figdir = check_exists_create_if_not(os.path.join(dir_in, 'figs')) # dir_in.split('/')[-1]+'.png'))
    #myplot = os.path.join(figdir, dir_in.split('/')[-1] + '.png')
    plt.savefig('cross_val.png')
    plt.show()

def __average_train_val_losses(log_collection):
    train_avgs, val_avgs = [], []
    
    for log_file in log_collection:
        loss, val_loss = __extract_losses(log_file)
        train_avgs.append(np.array(loss))
        val_avgs.append(np.array(val_loss))
    
    train_concat = list(reduce(lambda x1, x2: x1 + x2, train_avgs) / len(log_collection))
    val_concat = list(reduce(lambda x1, x2: x1 + x2, val_avgs) / len(log_collection))
    
    return train_concat, val_concat

def __extract_losses(log_file):
    """
    Utility function that extracts the progression of both training and validation losses from
    the given training log file.

    params:
        log_file - the training log file
    return:
        -> the lists of training and validation losses
    """

    with open(log_file, 'r') as log:
        log = log.readlines()
        loss = []
        val_loss = []
        for line in log[1:]:
            line = line.rstrip()
            line = line.split(',')
            loss.append(float(float(line[LOSS])))
            val_loss.append(float(line[VAL_LOSS]))
        return loss, val_loss

if __name__ == "__main__":
    paths = sys.argv[1:]
    if len(paths) < 2:
        print("Please enter at least two paths for comparison.")
        sys.exit()
    compare_losses(paths)
