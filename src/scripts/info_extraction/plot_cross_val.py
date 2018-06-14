from we_panic_utils.basic_utils.basics import check_exists_create_if_not
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm 

plt.switch_backend('agg')

LOSS = 1
VAL_LOSS = 3

def plot_multiple_losses(dir_in):
    """
    Given a directory within the run history archive, this method will extract all of
    the training log files and produce two plots for all folds of a single cross validation training.
    This method supports any number of folds, and will assign each fold a unique color and label.

    params:
        dir_in : the input experiment directory 
    """

    logs = [os.path.join(dir_in, log) for log in os.listdir(dir_in) if '.log' in log]
    logs.sort()
    plt.figure('Cross Validation Training Loss and Validation Loss')
    losses, val_losses = [], []
    for log in logs:
        print(log)
        loss, val_loss = __extract_losses(log)
        losses.append(loss)
        val_losses.append(val_loss)
    
    color=cm.rainbow(np.linspace(0,1,len(logs)))
    patches = [mpatches.Patch(color=color, label=("Fold " + str(i))) for i, color in enumerate(color)]

    plt.subplot(211)
    plt.title(dir_in.split('/')[-1] + ' Cross Validation')
    plt.ylabel('Training Loss')
    for i, (loss, c) in enumerate(zip(losses, color)):
        plt.plot([j for j in range(len(loss))], loss, c=c, label='Fold {}'.format(i))

    plt.legend(handles=patches)

    plt.subplot(212)
    plt.ylabel('Validation Loss')
    for val_loss, c in zip(val_losses, color):
        plt.plot([i for i in range(len(val_loss))], val_loss, c=c) 

    plt.tight_layout()
    
    figdir = check_exists_create_if_not(os.path.join(dir_in, 'figs')) # dir_in.split('/')[-1]+'.png'))
    myplot = os.path.join(figdir, dir_in.split('/')[-1] + '.png')
    plt.savefig(myplot)

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

if __name__ == '__main__':
    f = sys.argv[1]
    plot_multiple_losses(f)
