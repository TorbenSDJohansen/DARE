# Code for DARE

## Variables
For ease of use, define variables that are used in commands below:
```
set DATADIR=Z:\data_cropouts\Labels\DARE
set EXPDIR=Z:\faellesmappe\tsdj\DARE\experiments
set EVALDIR=Z:\faellesmappe\tsdj\DARE\eval-timmsn=0.2.5

set ALL_DATASETS_FLAT=cihvr death-certificates-1 death-certificates-2 funeral-records police-register-sheets-1 police-register-sheets-2 swedish-records-birth-dates swedish-records-death-dates
set ALL_DATASETS=(cihvr, death-certificates-1, death-certificates-2, funeral-records, police-register-sheets-1, police-register-sheets-2, swedish-records-birth-dates, swedish-records-death-dates)

set DDMYY_DATASETS_FLAT=death-certificates-1 death-certificates-2 police-register-sheets-1 police-register-sheets-2 swedish-records-birth-dates
set DDMYY_DATASETS=(death-certificates-1, death-certificates-2, police-register-sheets-1, police-register-sheets-2, swedish-records-birth-dates)

set DDM_DATASETS_FLAT=cihvr funeral-records swedish-records-death-dates
set DDM_DATASETS=(cihvr, funeral-records, swedish-records-death-dates)

set EVAL_BATCHSIZE=2048
```
**Note**: Need to change `DATA_DIR` to match where the DARE database is saved and `EXPDIR` and `EVALDIR` to desired paths for models and evaluation results, respectively.

## Train

### Single dataset models
```
set cexp=cihvr
python train.py --formatter legacy.dates_ddm --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR%

set cexp=death-certificates-1
python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR%

set cexp=death-certificates-2
python train.py --formatter legacy.dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR%

set cexp=funeral-records
python train.py --formatter legacy.dates_ddm -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR%

set cexp=police-register-sheets-1
python train.py --formatter legacy.dates_ddmyy --epochs 90 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR%

set cexp=police-register-sheets-2
python train.py --formatter legacy.dates_ddmyy --epochs 120 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR%

set cexp=swedish-records-birth-dates
python train.py --formatter legacy.dates_ddmyy --epochs 120 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR%

set cexp=swedish-records-death-dates
python train.py --formatter legacy.dates_ddm --epochs 120 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR%
```

### Multi dataset models
```
set cexp=full-ddmyyyy
python train.py --formatter legacy.dates_ddmyyyy --epochs 60 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %ALL_DATASETS_FLAT% --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR%

set cexp=split-ddmyy
python train.py --formatter legacy.dates_ddmyy --epochs 60 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %DDMYY_DATASETS_FLAT% --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR%

set cexp=split-ddm
python train.py --formatter legacy.dates_ddm --epochs 90 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %DDM_DATASETS_FLAT% --experiment %cexp% --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR%
```

## Evaluate

### Single dataset models
```
set cexp=cihvr
python evaluate.py --formatter legacy.dates_ddm --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=death-certificates-1
python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=death-certificates-2
python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=funeral-records
python evaluate.py --formatter legacy.dates_ddm --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=police-register-sheets-1
python evaluate.py --formatter legacy.dates_ddmyy --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=police-register-sheets-2
python evaluate.py --formatter legacy.dates_ddmyy --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=swedish-records-birth-dates
python evaluate.py --formatter legacy.dates_ddmyy --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=swedish-records-death-dates
python evaluate.py --formatter legacy.dates_ddm --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

### Full ddmyyyy
```
set cexp=full-ddmyyyy
python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp% --dataset %ALL_DATASETS_FLAT% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

for %i in %ALL_DATASETS% DO python evaluate.py --formatter legacy.dates_ddmyyyy --output %EVALDIR%\%i\%cexp% --dataset %i --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

### Split ddmyy
```
set cexp=split-ddmyy
python evaluate.py --formatter legacy.dates_ddmyy --output %EVALDIR%\%cexp%\%cexp% --dataset %DDMYY_DATASETS_FLAT% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yamll --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

for %i in %DDMYY_DATASETS% DO python evaluate.py --formatter legacy.dates_ddmyy --output %EVALDIR%\%i\%cexp% --dataset %i --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

### Split ddm
```
set cexp=split-ddm
python evaluate.py --formatter legacy.dates_ddm --output %EVALDIR%\%cexp%\%cexp% --dataset %DDM_DATASETS_FLAT% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

for %i in %DDM_DATASETS% DO python evaluate.py --formatter legacy.dates_ddm --output %EVALDIR%\%i\%cexp% --dataset %i --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

## Predict
Single test for now
```
set cexp=death-certificates-1
python predict.py --formatter legacy.dates_ddmyyyy --output Z:\faellesmappe\tsdj\DARE\pred\%cexp%\%cexp% --dataset %cexp% --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage
```
FIXME: Change `--output` above to not hardcoded location.

## Fine-tune DDMYY model on DC-1
```
set cexp=death-certificates-1

python train.py ^
--formatter legacy.dates_ddmyy ^
--output %EXPDIR% ^
--dataset %cexp% ^
--experiment %cexp%-finetuned-ddmyy ^
--config cfgs/efficientnetv2_s.yaml ^
--data_dir %DATADIR% ^
--initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar

python train.py ^
--lr 0.05 ^
--formatter legacy.dates_ddmyy ^
--output %EXPDIR% ^
--dataset %cexp% ^
--experiment %cexp%-finetuned-ddmyy-lr-0.05 ^
--config cfgs/efficientnetv2_s.yaml ^
--data_dir %DATADIR% ^
--initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar

```

## Linking exercise
Variables...
```
set link-cells=birthdate-1 birthdate-2 birthdate-3 birthdate-4 birthdate-5 birthdate-6 birthdate-7 birthdate-8 birthdate-9 birthdate-10 birthdate-11 birthdate-12 birthdate-13 birthdate-14 birthdate-15 birthdate-16 birthdate-17 birthdate-18 birthdate-19 birthdate-20 birthdate-21 birthdate-22 birthdate-23 birthdate-24 birthdate-25
set link-pred-output=Z:\faellesmappe\tsdj\DARE\pred\danish-census-linking
set link-train-output=Z:\faellesmappe\tsdj\DARE\experiments\linking
```
FIXME: Change output folders above to not hardcoded location.

Predict on just one cell for now:
```
python predict.py --checkpoint %EXPDIR%\split-ddmyy\last.pth.tar --output %link-pred-output%\round0 --dataset-cells birthdate-1 --formatter legacy.dates_ddmyy --dataset storage --config cfgs/efficientnetv2_s.yaml --data_dir "W:\BDADSharedData\Spanish Flu\Denmark\census1916" -b %EVAL_BATCHSIZE% --plots montage
```
For all cells:
```
python predict.py --checkpoint %EXPDIR%\split-ddmyy\last.pth.tar --output %link-pred-output%\round0 --dataset-cells %link-cells% --formatter legacy.dates_ddmyy --dataset storage --config cfgs/efficientnetv2_s.yaml --data_dir "W:\BDADSharedData\Spanish Flu\Denmark\census1916" -b %EVAL_BATCHSIZE% --plots montage
```
Train (here on earlier round 1 as illustrations)
```
python train.py --lr 0.0625 --experiment round1 --labels-subdir round1 --dataset-structure nested --dataset-cells %link-cells% --evalset train-split-0.05 --initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar --output %link-train-output% --formatter legacy.dates_ddmyy --dataset storage --config cfgs/efficientnetv2_s.yaml --data_dir "W:\BDADSharedData\Spanish Flu\Denmark\census1916"
```




