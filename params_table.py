# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:14:20 2022

@author: sa-tsdj
"""


import os

from typing import List

import yaml

import pandas as pd


ROOT = r'Z:\faellesmappe\tsdj\DARE'


def extract_params(files: List[str]):
    cfgs = {}
    
    for file in files:
        with open(file, 'r') as f: # pylint: disable=C0103
            cfg = yaml.safe_load(f)
            
            if len(cfgs) == 0:
                for k, v in cfg.items():
                    cfgs[k] = [v]
            else:
                for k, v in cfg.items():
                    cfgs[k].append(v)

    return cfgs


def main():
    models = [
        'cihvr',
        'death-certificates-1',
        'death-certificates-2',
        'funeral-records',
        'police-register-sheets-1',
        'police-register-sheets-2',
        'swedish-records-birth-dates',
        'swedish-records-death-dates',
        'full-ddmyyyy',
        'split-ddm',
        'split-ddmyy',
        ]
    cols ={
        'aa': 'RandAugment', 
        'batch_size': 'Batch size', 
        'clip_grad': 'Gradient clip value', 
        'drop': 'Dropout prob.', 
        'drop_path': 'Stochastic depth prob.', 
        'epochs': 'Epochs', 
        'lr': 'Learning rate', 
        'momentum': 'Momentum', 
        'reprob': 'Random erase prob.', 
        'smoothing': 'Label smoothing', 
        'weight_decay': 'Weight decay',
        }
    
    files = [os.path.join(ROOT, 'experiments', x, 'args.yaml') for x in models]
    cfgs = extract_params(files)
    
    table = pd.DataFrame(cfgs)
    table = table[cols.keys()]
    table = table.rename(columns=cols)
    table = table.replace({'rand-m7-mstd0.5-inc1': '$N=2,M=7$'})
    table.index = models
    table = table.drop_duplicates()
    table = table.T
    
    print('\n'.join(table.to_latex(escape=False).split('\n')[4:-2]), file=open('./param_tab.tex', 'w'))


if __name__ == '__main__':
    main()
