import sys, os
import pandas as pd
import matplotlib.pyplot as plt

def convert_csv_to_scatter(csv):
    df = pd.read_csv(csv)
    hr, rr = df['HEART_RATE_BPM'].values.tolist(), df['RESP_RATE_BR_PM'].values.tolist()

    return hr, rr

if __name__ == '__main__':
    D = sys.argv[1]
    
    cvsets = os.path.join(D, 'CVsets')
    fig, axarr = plt.subplots(1, sharex=True, sharey=True)
    axarr.set_title('training')
    axarr.set_ylabel('respiratory rate')

    for csv in os.listdir(cvsets):
        print(csv)
        csvfull = os.path.join(cvsets, csv)
        hr, rr = convert_csv_to_scatter(csvfull)
        
        if 'train' in csv:
            i = 0
        else:
            i = 0

        axarr.scatter(hr, rr, label=csv)
    
    axarr.legend()
    plt.savefig('hr-rr-trainval.pdf')
