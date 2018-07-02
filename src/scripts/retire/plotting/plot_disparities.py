"""
plot_disparities.py
-------------------
--- report the correlation between predicted/actual heart_rate/resp_rate in 3 different plots
 -- scatter : X = actual heart rate, Y = predicted heart rate
 -- scatter : X = actual resp rate , Y = predicted resp rate
 -- scatter : X = actual heart rate - predicted heart rate, Y = actual resp rate - predicted resp rate 
------------------
take a run_history directory and do the plots 
"""

import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy.stats import pearsonr
import sys, os
import numpy as np

from keras.models import load_model
from we_panic_utils.nn import Engine
from we_panic_utils.nn.processing import FrameProcessor
from we_panic_utils.basic_utils.basics import check_exists_create_if_not
plt.switch_backend('agg')
def parse_args():
    parser = argparse.ArgumentParser(description='plot disparities')

    parser.add_argument('directory',
                        type=str,
                        help='run_history directory in question (rel or absolute path)')

    return parser


def get_relative_path(dirname):
    """
    get the relative path to this directory inside of run_history
    """
    if 'run_history' in dirname:
        if os.path.isdir(dirname):
            return dirname.split('/')[1] # just under run_history

        else:
            raise NotADirectoryError("invalid dir: %s " % dirname)

    else:
        if os.path.isdir(os.path.join('run_history', dirname)):
            return dirname

        else:
            raise NotADirectoryError("invalid dir: %s " % os.path.join('run_history', dirname))
        


def determine_top_model(relpath):
    """
    determine the top mdel in this run history directory
    -- return the filepath of that model to load
    """
    run_hist_dir = os.path.join('run_history', relpath)
    cvresults = pd.read_csv(os.path.join(run_hist_dir, 'cvresults.csv'))

    min_loss = cvresults['loss'].min()
    best_row = cvresults[cvresults['loss'] == min_loss]
    
    model_str = 'CV_%d_%s.h5' % (best_row['model_idx'].values.tolist()[0],
                                 best_row['model_type'].values.tolist()[0])
    best_model = os.path.join('run_history',relpath,'models', model_str)

    return best_model, model_str



if __name__ == '__main__':
    args = parse_args().parse_args()
    
    directory = args.directory
    relpath = get_relative_path(directory)
    top_mod, model_relpath = determine_top_model(relpath)
    fold_number = int(model_relpath.split('_')[1])
    model_type = top_mod.split('_')[2][:-3]

    fp = FrameProcessor(greyscale_on=True)
    e = Engine(data='frames',
               model_type=model_type,
               csv='wepanic_collated_catalogue.csv',
               batch_size=14,
               epochs=1,
               frameproc=fp,
               train=False,
               test=True,
               input_shape=(60, 32, 32, 1),

               inputs=os.path.join('run_history',relpath),
               outputs=os.path.join('run_history',relpath),
               steps_per_epoch=500)

    #print(top_mod)
    e.model = load_model(top_mod)
    e.test_set = pd.read_csv(os.path.join('run_history',relpath,'CVsets','val%d.csv' % fold_number))  

    pred, loss = e.run()
    observed = e.test_set[['HEART_RATE_BPM', 'RESP_RATE_BR_PM']].values.tolist()
    print('obs: ', observed)
    print('pred: ', pred)
    
    pred = pred.reshape((int(pred.shape[0]/2), 4))
    pred_avg = list(map(lambda row: [np.mean([row[0], row[2]]),
                                     np.mean([row[1], row[3]])], pred))
    #print(pred_avg)
    pred_df = pd.DataFrame(pred_avg, columns=['HEART_RATE', 'RESP_RATE']) 
    print(pred_df)
    
    figsdir = check_exists_create_if_not(os.path.join('run_history',relpath,'figs'))
    

    # heart rate x, y
    hrate_actual = e.test_set['HEART_RATE_BPM'].values.tolist()
    hrate_pred = pred_df['HEART_RATE'].values.tolist()
    r, p = pearsonr(hrate_actual, hrate_pred)
    plt.scatter(hrate_actual, hrate_pred, label='$R=%0.4f$' % r)
    plt.xlabel('actual heart rate')
    plt.ylabel('predicted heart rate')
    plt.savefig(os.path.join(figsdir, 'hrate_corr.png'))
    plt.clf() 
    
    # resp rate x, y
    resprate_actual = e.test_set['RESP_RATE_BR_PM'].values.tolist()
    resprate_pred = pred_df['RESP_RATE'].values.tolist()
    r, p = pearsonr(resprate_actual, resprate_pred)
    plt.scatter(resprate_actual, resprate_pred, label='$R=%0.4f$' % r)
    plt.xlabel('actual respiratory rate')
    plt.ylabel('predicted respiratory rate')
    plt.savefig(os.path.join(figsdir, 'resprate_corr.png'))
    plt.clf()

    # actual - pred
    diff_hrate = np.array(hrate_actual) - np.array(hrate_pred)
    diff_pred = np.array(resprate_actual) - np.array(resprate_pred)
    plt.scatter(diff_hrate, diff_pred)
    plt.xlabel('actual - predicted heart rate')
    plt.ylabel('actual - predicted respiratory rate')
    
    plt.savefig(os.path.join(figsdir, 'diff.png'))
