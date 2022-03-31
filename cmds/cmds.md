# Code for DARE

## Variables
For ease of use, define variables that are used in commands below:
```
set DATADIR=Z:\data_cropouts\Labels\DARE
set EXPDIR=Z:\faellesmappe\tsdj\DARE\experiments
set EVALDIR=Z:\faellesmappe\tsdj\DARE\eval

set ALL_DATASETS_FLAT=cihvr death-certificates-1 death-certificates-2 funeral-records police-register-sheets-1 police-register-sheets-2 swedish-records-birth-dates swedish-records-death-dates
set ALL_DATASETS=(cihvr, death-certificates-1, death-certificates-2, funeral-records, police-register-sheets-1, police-register-sheets-2, swedish-records-birth-dates, swedish-records-death-dates)

set DDMYY_DATASETS_FLAT=death-certificates-1 death-certificates-2 police-register-sheets-1 police-register-sheets-2 swedish-records-birth-dates
set DDMYY_DATASETS=(death-certificates-1, death-certificates-2, police-register-sheets-1, police-register-sheets-2, swedish-records-birth-dates)

set DDM_DATASETS_FLAT=cihvr funeral-records swedish-records-death-dates
set DDM_DATASETS=(cihvr, funeral-records, swedish-records-death-dates)

set EVAL_BATCHSIZE=2048
```

## Train

### Single dataset models
```
set cexp=cihvr
python train.py --formatter dates_ddm --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%

set cexp=death-certificates-1
python train.py --formatter dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%

set cexp=death-certificates-2
python train.py --formatter dates_ddmyyyy --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%

set cexp=funeral-records
python train.py --formatter dates_ddm -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%

set cexp=police-register-sheets-1
python train.py --formatter dates_ddmyy --epochs 90 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%

set cexp=police-register-sheets-2
python train.py --formatter dates_ddmyy --epochs 120 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%

set cexp=swedish-records-birth-dates
python train.py --formatter dates_ddmyy --epochs 120 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%

set cexp=swedish-records-death-dates
python train.py --formatter dates_ddm --epochs 120 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %cexp% --experiment %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%
```

### Multi dataset models
```
set cexp=full-ddmyyyy
python train.py --formatter dates_ddmyyyy --epochs 60 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %ALL_DATASETS_FLAT% --experiment %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%

set cexp=split-ddmyy
python train.py --formatter dates_ddmyy --epochs 60 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %DDMYY_DATASETS_FLAT% --experiment %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%

set cexp=split-ddm
python train.py --formatter dates_ddm --epochs 90 --warmup-epochs 5 -b 308 --lr 0.6 --output %EXPDIR% --dataset %DDM_DATASETS_FLAT% --experiment %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%
```

## Evaluate

### Single dataset models
```
set cexp=cihvr
python evaluate.py --formatter dates_ddm --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=death-certificates-1
python evaluate.py --formatter dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=death-certificates-2
python evaluate.py --formatter dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=funeral-records
python evaluate.py --formatter dates_ddm --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=police-register-sheets-1
python evaluate.py --formatter dates_ddmyy --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=police-register-sheets-2
python evaluate.py --formatter dates_ddmyy --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=swedish-records-birth-dates
python evaluate.py --formatter dates_ddmyy --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=swedish-records-death-dates
python evaluate.py --formatter dates_ddm --output %EVALDIR%\%cexp%\%cexp% --dataset %cexp% --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

### Full ddmyyyy
```
set cexp=full-ddmyyyy
python evaluate.py --formatter dates_ddmyyyy --output %EVALDIR%\%cexp%\%cexp% --dataset %ALL_DATASETS_FLAT% --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

for %i in %ALL_DATASETS% DO python evaluate.py --formatter dates_ddmyyyy --output %EVALDIR%\%i\%cexp% --dataset %i --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

### Split ddmyy
```
set cexp=split-ddmyy
python evaluate.py --formatter dates_ddmyy --output %EVALDIR%\%cexp%\%cexp% --dataset %DDMYY_DATASETS_FLAT% --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

for %i in %DDMYY_DATASETS% DO python evaluate.py --formatter dates_ddmyy --output %EVALDIR%\%i\%cexp% --dataset %i --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

### Split ddm
```
set cexp=split-ddm
python evaluate.py --formatter dates_ddm --output %EVALDIR%\%cexp%\%cexp% --dataset %DDM_DATASETS_FLAT% --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

for %i in %DDM_DATASETS% DO python evaluate.py --formatter dates_ddm --output %EVALDIR%\%i\%cexp% --dataset %i --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

## Predict
Single test for now
```
set cexp=death-certificates-1
python predict.py --formatter dates_ddmyyyy --output Z:\faellesmappe\tsdj\DARE\pred\%cexp%\%cexp% --dataset %cexp% --experiment %cexp% --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage
```

## Fine-tune DDMYY model on DC-1
```
set cexp=death-certificates-1

python train.py ^
--formatter dates_ddmyy ^
--output %EXPDIR% ^
--dataset %cexp% ^
--experiment %cexp%-finetuned-ddmyy ^
--config %EXPDIR%\cfgs\default.yaml ^
--data_dir %DATADIR% ^
--initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar

python train.py ^
--lr 0.05 ^
--formatter dates_ddmyy ^
--output %EXPDIR% ^
--dataset %cexp% ^
--experiment %cexp%-finetuned-ddmyy-lr-0.05 ^
--config %EXPDIR%\cfgs\default.yaml ^
--data_dir %DATADIR% ^
--initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar

```

## Transfer learn on ATLASS
Note relatively square image size often the case. Use 224x224

### Tune LR
Search for LR. NOTE: 2xGPU training. log2uniform distr.
```
set cexp=no_empty

for %i in (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625) DO ^
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr %i ^
--input-size 3 224 224 ^
--experiment %cexp%-tl-lr-%i ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--evalset train-split-0.1 ^
--config %EXPDIR%\cfgs\default.yaml

for %i in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125) DO ^
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr %i ^
--input-size 3 224 224 ^
--experiment %cexp%-lr-%i ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--evalset train-split-0.1 ^
--config %EXPDIR%\cfgs\default.yaml

for %i in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125) DO ^
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr %i ^
--input-size 3 224 224 ^
--experiment %cexp%-no-pretrain-lr-%i ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--evalset train-split-0.1 ^
--config %EXPDIR%\cfgs\default-no-pretrained.yaml

```

### Train
```
set cexp=no_empty

python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 1.0 ^
--input-size 3 224 224 ^
--experiment %cexp%-tl ^
--output %EXPDIR%\atlass ^
--formatter dates_ddmyy ^
--initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml

python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 0.25 ^
--input-size 3 224 224 ^
--experiment %cexp% ^
--output %EXPDIR%\atlass ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml

python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 0.5 ^
--input-size 3 224 224 ^
--experiment %cexp%-no-pretrain ^
--output %EXPDIR%\atlass ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default-no-pretrained.yaml

```

### Evaluate

## Transfer learn on Danish Census
Note use of same input-size, seems ratio is OK

### Tune LR
Search for LR. NOTE: 2xGPU training. log2uniform distr.
(1) Small TL
(2) Small non-TL
(3) Large TL
(4) Large non-TL
```
set cexp=danish-census-small
for %i in (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625) DO python -m torch.distributed.launch --nproc_per_node=2 train.py --lr %i --experiment %i --output %EXPDIR%\danish-census\tl-exp-%cexp% --evalset train-split-0.05 --formatter dates_ddmyy --initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%
for %i in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125) DO python -m torch.distributed.launch --nproc_per_node=2 train.py --lr %i --experiment %i-non-tl --output %EXPDIR%\danish-census\tl-exp-%cexp% --evalset train-split-0.05 --formatter dates_ddmyy --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%

set cexp=danish-census-large
for %i in (0.5, 0.25, 0.125, 0.0625, 0.03125) DO python -m torch.distributed.launch --nproc_per_node=2 train.py --lr %i --experiment %i --output %EXPDIR%\danish-census\tl-exp-%cexp% --evalset train-split-0.05 --formatter dates_ddmyy --initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%
for %i in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25) DO python -m torch.distributed.launch --nproc_per_node=2 train.py --lr %i --experiment %i-non-tl --output %EXPDIR%\danish-census\tl-exp-%cexp% --evalset train-split-0.05 --formatter dates_ddmyy --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%
```

### Train
(1) Small TL
(2) Small non-TL
(3) Large TL
(4) Large non-TL
```
set cexp=danish-census-small
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 0.0625 --experiment %cexp%-tl --output %EXPDIR%\danish-census --formatter dates_ddmyy --initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 4.0 --experiment %cexp% --output %EXPDIR%\danish-census --formatter dates_ddmyy --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%

set cexp=danish-census-large
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 0.25 --experiment %cexp%-tl --output %EXPDIR%\danish-census --formatter dates_ddmyy --initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 4.0 --experiment %cexp% --output %EXPDIR%\danish-census --formatter dates_ddmyy --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR%
```

### Evaluate
```
set cexp=danish-census-small
python evaluate.py --output %EVALDIR%\%cexp%\%cexp%-tl --checkpoint %EXPDIR%\danish-census\%cexp%-tl\last.pth.tar --formatter dates_ddmyy --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
python evaluate.py --output %EVALDIR%\%cexp%\%cexp% --checkpoint %EXPDIR%\danish-census\%cexp%\last.pth.tar --formatter dates_ddmyy --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc

set cexp=danish-census-large
python evaluate.py --output %EVALDIR%\%cexp%\%cexp%-tl --checkpoint %EXPDIR%\danish-census\%cexp%-tl\last.pth.tar --formatter dates_ddmyy --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
python evaluate.py --output %EVALDIR%\%cexp%\%cexp% --checkpoint %EXPDIR%\danish-census\%cexp%\last.pth.tar --formatter dates_ddmyy --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

For "fun", evaluate directly from split-ddmyy
```
set cexp=split-ddmyy
python evaluate.py --formatter dates_ddmyy --output %EVALDIR%\danish-census-large\%cexp% --dataset danish-census-large --checkpoint %EXPDIR%\%cexp%\last.pth.tar --config %EXPDIR%\cfgs\default.yaml --data_dir %DATADIR% -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

## Transfer learn on SVHN

### Tune LR
**Settings**:
1. Image size: 256x256.
1. "Goodfellow paper crop".
1. Not using extra. 
1. Using full train.
1. Cut head for TL.

```
for %i in (4.0, 2.0, 1.0, 0.5, 0.25, 0.125) DO python -m torch.distributed.launch --nproc_per_node=2 train.py --lr %i --experiment lr-%i --evalset train-split-0.05 --input-size 3 256 256 --output %EXPDIR%\svhn\lr-exp --formatter svhn_as_numseq --dataset svhn --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test
for %i in (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625) DO python -m torch.distributed.launch --nproc_per_node=2 train.py --lr %i --experiment tl-lr-%i --evalset train-split-0.05 --input-size 3 256 256 --output %EXPDIR%\svhn\lr-exp --formatter svhn_as_numseq --dataset svhn --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test --initial-checkpoint %EXPDIR%\full-ddmyyyy\last.pth.tar --drop-modules classifier*

```

As sanity check (~96.82%): `python evaluate.py --output %EVALDIR%\svhn\lr-0.5 --checkpoint Z:\faellesmappe\tsdj\DARE\experiments\svhn\lr-exp\lr-0.5\last.pth.tar --formatter svhn_as_numseq --dataset svhn --input-size 3 256 256 --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation`

Now for the small 10k train sample
```
for %i in (4.0, 2.0, 1.0, 0.5, 0.25, 0.125) DO python -m torch.distributed.launch --nproc_per_node=2 train.py --lr %i --experiment lr-%i --evalset train-split-0.05 --input-size 3 256 256 --output %EXPDIR%\svhn\lr-exp-small --formatter svhn_as_numseq --dataset svhn_small --config %EXPDIR%\cfgs\default.yaml --data_dir Z:\data_cropouts --dataset-cells train
for %i in (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625) DO python -m torch.distributed.launch --nproc_per_node=2 train.py --lr %i --experiment tl-lr-%i --evalset train-split-0.05 --input-size 3 256 256 --output %EXPDIR%\svhn\lr-exp-small --formatter svhn_as_numseq --dataset svhn_small --config %EXPDIR%\cfgs\default.yaml --data_dir Z:\data_cropouts --dataset-cells train --initial-checkpoint %EXPDIR%\full-ddmyyyy\last.pth.tar --drop-modules classifier*

```

Sanity check:
```
for %i in (4.0, 2.0, 1.0, 0.5, 0.25, 0.125) DO python evaluate.py --input-size 3 256 256 --output %EVALDIR%\svhn\lr-%i --formatter svhn_as_numseq --dataset svhn_small --config %EXPDIR%\cfgs\default.yaml --data_dir Z:\data_cropouts --checkpoint %EXPDIR%\svhn\lr-exp-small\lr-%i\last.pth.tar
for %i in (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625) DO python evaluate.py --input-size 3 256 256 --output %EVALDIR%\svhn\tl-lr-%i --formatter svhn_as_numseq --dataset svhn_small --config %EXPDIR%\cfgs\default.yaml --data_dir Z:\data_cropouts --checkpoint %EXPDIR%\svhn\lr-exp-small\tl-lr-%i\last.pth.tar

```

### Train
```
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 0.5 --experiment svhn-lr-0.5 --input-size 3 256 256 --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset svhn --config %EXPDIR%\cfgs\default.yaml --data_dir Z:\data_cropouts --dataset-cells train --dataset-cells-eval test --initial-log
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 0.25 --experiment svhn-tl-lr-0.25 --input-size 3 256 256 --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset svhn --config %EXPDIR%\cfgs\default.yaml --data_dir Z:\data_cropouts --dataset-cells train --dataset-cells-eval test --initial-checkpoint %EXPDIR%\full-ddmyyyy\last.pth.tar --drop-modules classifier*  --initial-log

```

Now for the small 10k train sample (weirdly high LR for TL...)
```
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 1.0 --experiment svhn-small-lr-1.0 --input-size 3 256 256 --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset svhn_small --config %EXPDIR%\cfgs\default.yaml --data_dir Z:\data_cropouts --dataset-cells train --dataset-cells-eval test --initial-log
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 2.0 --experiment svhn-small-tl-lr-2.0 --input-size 3 256 256 --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset svhn_small --config %EXPDIR%\cfgs\default.yaml --data_dir Z:\data_cropouts --dataset-cells train --dataset-cells-eval test --initial-checkpoint %EXPDIR%\full-ddmyyyy\last.pth.tar --drop-modules classifier*  --initial-log

```

### Evaluate
```
python evaluate.py --input-size 3 256 256 --output %EVALDIR%\svhn\svhn --formatter svhn_as_numseq --dataset svhn --config %EXPDIR%\cfgs\default.yaml --data_dir Z:\data_cropouts --plots montage cov-acc cer-acc --checkpoint %EXPDIR%\svhn\svhn-lr-0.5\last.pth.tar
python evaluate.py --input-size 3 256 256 --output %EVALDIR%\svhn\svhn-tl --formatter svhn_as_numseq --dataset svhn --config %EXPDIR%\cfgs\default.yaml --data_dir Z:\data_cropouts --plots montage cov-acc cer-acc --checkpoint %EXPDIR%\svhn\svhn-tl-lr-0.25\last.pth.tar

```

Now for the models trained on the small 10k train sample
```
python evaluate.py --input-size 3 256 256 --output %EVALDIR%\svhn\svhn-small --formatter svhn_as_numseq --dataset svhn_small --config %EXPDIR%\cfgs\default.yaml --data_dir Z:\data_cropouts --plots montage cov-acc cer-acc --checkpoint %EXPDIR%\svhn\svhn-small-lr-1.0\last.pth.tar
python evaluate.py --input-size 3 256 256 --output %EVALDIR%\svhn\svhn-small-tl --formatter svhn_as_numseq --dataset svhn_small --config %EXPDIR%\cfgs\default.yaml --data_dir Z:\data_cropouts --plots montage cov-acc cer-acc --checkpoint %EXPDIR%\svhn\svhn-small-tl-lr-2.0\last.pth.tar

```


### Random experiments
**TODO** These are now in subfolder rand-exp. Need to add to --output.
However, these were only meant as initial checks, so no real need to update anything.

```
set cexp=svhn
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 1.0 --experiment lr-1.0 --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 2.0 --experiment lr-2.0 --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 4.0 --experiment lr-4.0 --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test

python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 1.0 --experiment lr-1.0-repl --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test

python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 1.0 --experiment 224x224-lr-1.0 --input-size 3 224 224 --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 1.0 --experiment 237x237-lr-1.0 --input-size 3 237 237 --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 1.0 --experiment 256x256-lr-1.0 --input-size 3 256 256 --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test

python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 1.0 --experiment dateformat-lr-1.0 --output %EXPDIR%\svhn --formatter svhn_as_date --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test

python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 1.0 --experiment lr-1.0-with-extra --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train extra --dataset-cells-eval test

python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 2.5 --experiment tl-lr-2.5 --output %EXPDIR%\svhn --formatter svhn_as_date --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test --initial-checkpoint %EXPDIR%\full-ddmyyyy\last.pth.tar
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 1.0 --experiment tl-lr-1.0 --output %EXPDIR%\svhn --formatter svhn_as_date --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test --initial-checkpoint %EXPDIR%\full-ddmyyyy\last.pth.tar
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 0.25 --experiment tl-lr-0.25 --output %EXPDIR%\svhn --formatter svhn_as_date --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test --initial-checkpoint %EXPDIR%\full-ddmyyyy\last.pth.tar
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 0.025 --experiment tl-lr-0.025 --output %EXPDIR%\svhn --formatter svhn_as_date --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test --initial-checkpoint %EXPDIR%\full-ddmyyyy\last.pth.tar

python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 1.0 --experiment tl-cut-head-lr-1.0 --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test --initial-checkpoint %EXPDIR%\full-ddmyyyy\last.pth.tar --drop-modules classifier*
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 0.25 --experiment tl-cut-head-lr-0.25 --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test --initial-checkpoint %EXPDIR%\full-ddmyyyy\last.pth.tar --drop-modules classifier*
python -m torch.distributed.launch --nproc_per_node=2 train.py --lr 0.025 --experiment tl-cut-head-lr-0.025 --output %EXPDIR%\svhn --formatter svhn_as_numseq --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --dataset-cells train --dataset-cells-eval test --initial-checkpoint %EXPDIR%\full-ddmyyyy\last.pth.tar --drop-modules classifier*

python evaluate.py --output %EVALDIR%\%cexp%\%cexp% --formatter svhn_as_numseq --dataset %cexp% --config %EXPDIR%\cfgs\default.yaml --data_dir W:\BDADSharedData\FilesForInstallation --checkpoint %EXPDIR%\%cexp%\last.pth.tar -b %EVAL_BATCHSIZE% --plots montage cov-acc cer-acc
```

## Linking exercise
Variables...
```
set link-cells=birthdate-1 birthdate-2 birthdate-3 birthdate-4 birthdate-5 birthdate-6 birthdate-7 birthdate-8 birthdate-9 birthdate-10 birthdate-11 birthdate-12 birthdate-13 birthdate-14 birthdate-15 birthdate-16 birthdate-17 birthdate-18 birthdate-19 birthdate-20 birthdate-21 birthdate-22 birthdate-23 birthdate-24 birthdate-25
set link-pred-output=Z:\faellesmappe\tsdj\DARE\pred\danish-census-linking
set link-train-output=Z:\faellesmappe\tsdj\DARE\experiments\linking
```

Predict on just one cell for now:
```
python predict.py --checkpoint %EXPDIR%\split-ddmyy\last.pth.tar --output %link-pred-output%\round0 --dataset-cells birthdate-1 --formatter dates_ddmyy --dataset storage --config %EXPDIR%\cfgs\default.yaml --data_dir "W:\BDADSharedData\Spanish Flu\Denmark\census1916" -b %EVAL_BATCHSIZE% --plots montage
```
For all cells:
```
python predict.py --checkpoint %EXPDIR%\split-ddmyy\last.pth.tar --output %link-pred-output%\round0 --dataset-cells %link-cells% --formatter dates_ddmyy --dataset storage --config %EXPDIR%\cfgs\default.yaml --data_dir "W:\BDADSharedData\Spanish Flu\Denmark\census1916" -b %EVAL_BATCHSIZE% --plots montage
```
Train (here on earlier round 1 as illustrations)
```
python train.py --lr 0.0625 --experiment round1 --labels-subdir round1 --dataset-structure nested --dataset-cells %link-cells% --evalset train-split-0.05 --initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar --output %link-train-output% --formatter dates_ddmyy --dataset storage --config %EXPDIR%\cfgs\default.yaml --data_dir "W:\BDADSharedData\Spanish Flu\Denmark\census1916"
```




