"""
build a directory of the top k models from their common
parent directory
"""

import argparse
import os
from glob import glob
import pandas as pd
from shutil import copyfile

def parse_args():

    parser = argparse.ArgumentParser(description='build a directory of top k models')
    parser.add_argument('parent_directory',
                        type=str,
                        help='common parent directory; usually something like run_history/')

    parser.add_argument('k',
                        type=int,
                        help='number of models to collect')

    parser.add_argument('--output', '-o',
                        type=str,
                        help='directory to copy the models',
                        default=None)


    return parser

def validate_args(args):
    
    if not os.path.exists(args.parent_directory):
        raise FileNotFoundError('Parent directory does not exist: {}'.format(args.parent_directory))

    if not os.path.isdir(args.parent_directory):
        raise NotADirectoryError('Not a directory : {}'.format(args.parent_directory))
    
    if not 0 < args.k:
        raise ValueError('Top K obviously should be > 0, what am I supposed to do with {}'.format(args.k))

    args.output = next_available_output(args)
    return args


def next_available_output(args):

    k = args.k
    output = args.output

    if output is None:
        if os.path.isdir('committee_top{}'.format(k)):
            i = 0
            while os.path.isdir('committee{}_top{}'.format(i,k)):
                i += 1

            com =  'committee{}_top{}'.format(i, k)

        else:
            com = 'committee_top{}'.format(k)

        os.makedirs(com)
        print('Generated an output directory called {}'.format(com))
        return com

    else:
        com = output
        if not os.path.exists(com):
            os.makedirs(com, exist_ok=True)
            print('Generated an output directory called {}'.format(com))
        else:
            print(f'Copying files into {com}')

        return com

if __name__ == '__main__':

    args = validate_args(parse_args().parse_args())
    
    all_cvresults = glob(os.path.join(args.parent_directory, '*', 'cvresults.csv'))
    all_cvdfs = [pd.read_csv(cvr) for cvr in all_cvresults]
    
    # compile each cvresults.csv
    compiled_df = pd.DataFrame(columns=['model_path','run_directory','loss','fold'])
    
    i = 0
    for cvr_pth, cvdf in zip(all_cvresults, all_cvdfs):
        rundir = "/".join(cvr_pth.split('/')[:-1])
        models_pth = os.path.join(rundir, 'models')
        assert os.path.isdir(models_pth), 'Error occurred finding {}'.format(models_pth)
        
        for idx, row in cvdf.iterrows():
            loss = row['loss']
            fold_number = row['model_idx']
            model_type  = row['model_type']
            
            model_pth = os.path.join(models_pth, 'CV_%d_%s.h5' % (fold_number, model_type))
            compiled_df.loc[i] = [model_pth, rundir, loss, fold_number]
            i += 1

    compiled_df.sort_values(by=['loss'], inplace=True, ascending=True)
        
    # find top K losses
    topKdf = pd.DataFrame(compiled_df.values.tolist()[:args.k], columns=compiled_df.columns)
    
    # translate model paths ----> committee member names
    slug = "{rundir}__Fold_{fold}__Loss_{loss:4.03f}.h5"
    
    #print('Copying {} files into {}'.format(args.k, args.output))
    for idx, row in topKdf.iterrows():
        rundir = row['run_directory'].split('/')[-1]
        loss = row['loss']
         
        # copy that list of files in to the output directory
        source = row['model_path']
        target = os.path.join(args.output, slug.format(rundir=rundir,loss=loss, fold=row['fold']))
        print(source, target, sep=' --> ')        
        copyfile(source, target)
    
    print(':)')
