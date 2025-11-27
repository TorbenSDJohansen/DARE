## Variables
For ease of use, define variables that are used in commands below:
```
set DATADIR=Z:\data_cropouts\Labels\DARE
set EXPDIR=Z:\faellesmappe\tsdj\DARE\ablations
set EVALDIR=Z:\faellesmappe\tsdj\DARE\eval-ablations

set ALL_DATASETS_FLAT=cihvr death-certificates-1 death-certificates-2 funeral-records police-register-sheets-1 police-register-sheets-2 swedish-records-birth-dates swedish-records-death-dates
set ALL_DATASETS=(cihvr, death-certificates-1, death-certificates-2, funeral-records, police-register-sheets-1, police-register-sheets-2, swedish-records-birth-dates, swedish-records-death-dates)

set DDMYY_DATASETS_FLAT=death-certificates-1 death-certificates-2 police-register-sheets-1 police-register-sheets-2 swedish-records-birth-dates
set DDMYY_DATASETS=(death-certificates-1, death-certificates-2, police-register-sheets-1, police-register-sheets-2, swedish-records-birth-dates)

set DDM_DATASETS_FLAT=cihvr funeral-records swedish-records-death-dates
set DDM_DATASETS=(cihvr, funeral-records, swedish-records-death-dates)

set EVAL_BATCHSIZE=2048
```
**Note**: Need to change `DATA_DIR` to match where the DARE database is stored and `EXPDIR` and `EVALDIR` to desired paths for models and evaluation results, respectively.


# Train

## DC-1

```
set cexp=death-certificates-1

python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp%-low-reg --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --drop 0.0 --drop-path 0.0 --weight-decay 0.0 --initial-log --log-wandb

python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp%-no-ls --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --smoothing 0.0 --initial-log --log-wandb

python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp%-no-reg --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --drop 0.0 --drop-path 0.0 --weight-decay 0.0 --smoothing 0.0 --initial-log --log-wandb

python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp%-no-aug --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --no-aug --initial-log --log-wandb

python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp%-no-aug-reg --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --no-aug --drop 0.0 --drop-path 0.0 --weight-decay 0.0 --smoothing 0.0 --initial-log --log-wandb

python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp%-lr=2.0 --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --lr 2.0 --initial-log --log-wandb

python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp%-lr=1.0 --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --lr 1.0 --initial-log --log-wandb

python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp%-lr=0.25 --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --lr 0.25 --initial-log --log-wandb

python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp%-lr=0.125 --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --lr 0.125 --initial-log --log-wandb

python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp%-no-sched --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --sched None --initial-log --log-wandb
```

## PR-2

```
set cexp=police-register-sheets-2

python train.py --formatter legacy.dates_ddmyy --epochs 120 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp%-no-aug-reg --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --no-aug --drop 0.0 --drop-path 0.0 --weight-decay 0.0 --smoothing 0.0 --initial-log --log-wandb --read-from-tar

python train.py --formatter legacy.dates_ddmyy --epochs 120 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp%-no-reg --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --drop 0.0 --drop-path 0.0 --weight-decay 0.0 --smoothing 0.0 --initial-log --log-wandb --read-from-tar

python train.py --formatter legacy.dates_ddmyy --epochs 120 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp%-low-reg --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --drop 0.0 --drop-path 0.0 --weight-decay 0.0 --initial-log --log-wandb --read-from-tar

python train.py --formatter legacy.dates_ddmyy --epochs 120 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp%-no-aug --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --no-aug --initial-log --log-wandb --read-from-tar
```

## DDMYYYY

```
set cexp=full-ddmyyyy

python train.py --formatter legacy.dates_ddmyyyy --epochs 60 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %ALL_DATASETS_FLAT% --experiment %cexp%-no-aug-reg --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --no-aug --drop 0.0 --drop-path 0.0 --weight-decay 0.0 --smoothing 0.0 --initial-log --log-wandb
```

# Evaluate

## DC-1

```
set cexp=death-certificates-1

python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp%-low-reg --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-low-reg\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp%-no-ls --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-no-ls\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp%-no-reg --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-no-reg\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp%-no-aug --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-no-aug\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp%-no-aug-reg --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-no-aug-reg\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp%-lr=2.0 --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-lr=2.0\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp%-lr=1.0 --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-lr=1.0\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp%-lr=0.25 --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-lr=0.25\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp%-lr=0.125 --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-lr=0.125\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp%-no-sched --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-no-sched\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

## PR-2

```
set cexp=police-register-sheets-2

python evaluate.py --formatter legacy.dates_ddmyy --output %EVALDIR%\%cexp%\%cexp%-no-reg --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-no-reg\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

python evaluate.py --formatter legacy.dates_ddmyy --output %EVALDIR%\%cexp%\%cexp%-low-reg --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-low-reg\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

python evaluate.py --formatter legacy.dates_ddmyy --output %EVALDIR%\%cexp%\%cexp%-no-aug --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-no-aug\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

python evaluate.py --formatter legacy.dates_ddmyy --output %EVALDIR%\%cexp%\%cexp%-no-aug-reg --dataset %cexp% --checkpoint %EXPDIR%\%cexp%-no-aug-reg\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

## DDMYYYY

```
set cexp=full-ddmyyyy
python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp%-no-aug-reg --dataset %ALL_DATASETS_FLAT% --checkpoint %EXPDIR%\%cexp%-no-aug-reg\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

for %i in %ALL_DATASETS% DO python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%i\%cexp%-no-aug-reg --dataset %i --checkpoint %EXPDIR%\%cexp%-no-aug-reg\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```