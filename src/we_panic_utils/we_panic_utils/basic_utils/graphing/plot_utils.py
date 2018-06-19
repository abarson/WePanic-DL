import os
import pandas as pd
import numpy as np

from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm 
plt.switch_backend('agg')

from we_panic_utils.basic_utils.basics import check_exists_create_if_not

LOSS = 1
VAL_LOSS = 3

def plot_multiple_losses(dir_in, dir_out=None):
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
    
    if dir_out is None:
        figdir = check_exists_create_if_not(os.path.join(dir_in, 'figs')) # dir_in.split('/')[-1]+'.png'))
        myplot = os.path.join(figdir, dir_in.split('/')[-1] + '.png')
    else:
        check_exists_create_if_not(os.path.join(dir_out))
        myplot = os.path.join(dir_out, dir_in.split('/')[-1] + '.png')
    plt.savefig(myplot)

def compare_losses(directories, dir_out):
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
    
    all_models_str = ""
    for directory in directories:
        all_models_str += directory.split('/')[-1] + '+'
    all_models_str = all_models_str.strip('+')
    all_models_str += '.png'
    
    figdir = check_exists_create_if_not(os.path.join(dir_out)) # dir_in.split('/')[-1]+'.png'))
    myplot = os.path.join(figdir, all_models_str)
    plt.savefig(myplot)

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
