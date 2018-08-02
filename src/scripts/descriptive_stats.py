"""
report descriptive statistics and generate some plots about the data
"""
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sys, os
import numpy as np
import argparse

plt.switch_backend('agg')


def parse_args():
    pass


def mean_std(df, col):
    return df[col].mean(), df[col].std()

if __name__ == '__main__':
    # assumes the most up to date
    wpc_cc = sys.argv[1]
    
    figs_dir = 'figs'
    now = datetime.datetime.now().strftime('%d-%m-%Y')
    param_slug = 'descriptive_stats__{}'.format(now)
    
    output_file = param_slug + '.png'
    
    
    wpc_df = pd.read_csv(wpc_cc)
    wpc_valid = wpc_df[wpc_df['GOOD'] > 0]
    wpc_train = wpc_valid[wpc_valid['GOOD'] == 1]
    wpc_test = wpc_valid[wpc_valid['GOOD'] == 2]

    heart_rates = wpc_valid['HEART_RATE_BPM'].values.tolist()
    resp_rates = wpc_valid['RESP_RATE_BR_PM'].values.tolist()
    
    print('# training samples: {}'.format(len(wpc_train)))
    print('# testing samples:  {}'.format(len(wpc_test)))
    
    mean_hr, std_hr = mean_std(wpc_valid, 'HEART_RATE_BPM')
    mean_rr, std_rr = mean_std(wpc_valid, 'RESP_RATE_BR_PM')

    print('  all samples: mean_hr={}, std_hr={}, mean_rr={}, std_rr={}'.format(mean_hr, std_hr, mean_rr, std_rr))

    mean_hr, std_hr = mean_std(wpc_train, 'HEART_RATE_BPM')
    mean_rr, std_rr = mean_std(wpc_train, 'RESP_RATE_BR_PM')

    print('train samples: mean_hr={}, std_hr={}, mean_rr={}, std_rr={}'.format(mean_hr, std_hr, mean_rr, std_rr))

    mean_hr, std_hr = mean_std(wpc_test, 'HEART_RATE_BPM')
    mean_rr, std_rr = mean_std(wpc_test, 'RESP_RATE_BR_PM')

    print(' test samples: mean_hr={}, std_hr={}, mean_rr={}, std_rr={}'.format(mean_hr, std_hr, mean_rr, std_rr))

    hr_test, rr_test = wpc_test['HEART_RATE_BPM'].values.tolist(), wpc_test['RESP_RATE_BR_PM'].values.tolist()
    hr_train, rr_train = wpc_train['HEART_RATE_BPM'].values.tolist(), wpc_train['RESP_RATE_BR_PM'].values.tolist()

    plt.scatter(hr_train, rr_train, label='training data')
    plt.scatter(hr_test, rr_test, label='testing data')
    plt.xlabel('heart rate (BPM)')
    plt.ylabel('respiratory rate (BR PM)')
    plt.legend()
    plt.savefig(os.path.join(figs_dir, output_file))
