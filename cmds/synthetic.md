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

```
set cexp=synthetic

python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp%-11627 --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --initial-log --log-wandb
```

## Transfer learn to DC-1

```
set cexp=death-certificates-1

python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --initial-checkpoint %EXPDIR%\synthetic\last.pth.tar --initial-log --log-wandb
```

# Evaluate

## Synthetic Model

```
set cexp=death-certificates-1

python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\synthetic --dataset %cexp% --checkpoint %EXPDIR%\synthetic\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

## DC-1

```
set cexp=death-certificates-1

python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```