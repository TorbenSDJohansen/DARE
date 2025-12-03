## Variables
For ease of use, define variables that are used in commands below:
```
set DATADIR=Z:\data_cropouts\Labels\DARE
set EXPDIR=Z:\faellesmappe\tsdj\DARE\synthetic
set EVALDIR=Z:\faellesmappe\tsdj\DARE\eval-synthetic

set EVAL_BATCHSIZE=2048
```

# Train

## Pre-train on synthetic data

Version(s) not first pre-trained on ImageNet21k, with formatter suitable for ATLASS dates

```
set cexp=synthetic

python train.py --formatter legacy.dates_ddmyy --output %EXPDIR% --dataset %cexp%-11627 --experiment %cexp%-no-pretrain --config Z:\faellesmappe\tsdj\DARE\experiments\cfgs\default-no-pretrained.yaml --data_dir %DATADIR% --initial-log --log-wandb

python train.py --formatter legacy.dates_ddmyy --output %EXPDIR% --dataset %cexp%-11627 --experiment %cexp%-no-pretrain-224x224 --config Z:\faellesmappe\tsdj\DARE\experiments\cfgs\default-no-pretrained.yaml --input-size 3 224 224 --data_dir %DATADIR% --initial-log --log-wandb
```

## Transfer learn to ATLASS

This allow us to compare against the large host of models we have trained on ATLASS.
For example, perhaps synthetic data is a better starting point than from scratch, but worse than ImageNet21k.
Or perhaps better than ImageNet21k but worse than DARE.

```
set cexp=atlass

python train.py --formatter legacy.dates_ddmyy --output %EXPDIR% --dataset atlass --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --input-size 3 224 224 --data_dir %DATADIR% --initial-checkpoint %EXPDIR%\synthetic-no-pretrain\last.pth.tar --initial-log --log-wandb

python train.py --formatter legacy.dates_ddmyy --output %EXPDIR% --dataset atlass --experiment %cexp%-from-224x224 --config cfgs/efficientnetv2_s.yaml --input-size 3 224 224 --data_dir %DATADIR% --initial-checkpoint %EXPDIR%\synthetic-no-pretrain-224x224\last.pth.tar --initial-log --log-wandb
```

# Evaluate

## ATLASS

```
set cexp=atlass

python evaluate.py --formatter legacy.dates_ddmyy --output %EVALDIR%\%cexp%\synthetic-pretrain-from-224x224 --dataset %cexp% --input-size 3 224 224 --checkpoint %EXPDIR%\%cexp%-from-224x224\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```
