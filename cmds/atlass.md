# Code for DARE

## Variables
For ease of use, define variables that are used in commands below:
```
set DATADIR=Z:\data_cropouts\Labels\DARE
set EXPDIR=Z:\faellesmappe\tsdj\DARE\experiments
set EVALDIR=Z:\faellesmappe\tsdj\DARE\eval

set EVAL_BATCHSIZE=2048
```

## Transfer learn on ATLASS
Note relatively square image size often the case. Use 224x224

### Test nb. epochs
```
for %e in (250, 1000, 2500) DO ^
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 0.5 ^
--epochs %e ^
--input-size 3 224 224 ^
--experiment %cexp%-tl-epoch-%e ^
--output %EXPDIR%\atlass\epoch-search ^
--formatter dates_ddmyy ^
--initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml

for %e in (250, 1000, 2500) DO ^
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 0.5 ^
--epochs %e ^
--input-size 3 224 224 ^
--experiment %cexp%-epoch-%e ^
--output %EXPDIR%\atlass\epoch-search ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml

for %e in (250, 1000, 2500) DO ^
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 0.5 ^
--epochs %e ^
--input-size 3 224 224 ^
--experiment %cexp%-no-pretrain-epoch-%e ^
--output %EXPDIR%\atlass\epoch-search ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default-no-pretrained.yaml

```

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
--config %EXPDIR%\cfgs\default.yaml

for %i in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125) DO ^
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr %i ^
--epochs 1000 ^
--input-size 3 224 224 ^
--experiment %cexp%-lr-%i ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml

for %i in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125) DO ^
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr %i ^
--epochs 1000 ^
--input-size 3 224 224 ^
--experiment %cexp%-no-pretrain-lr-%i ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default-no-pretrained.yaml

```

Test also 250 epochs performance of models not TL from DARE (performance @ same computation).
```
set cexp=no_empty

for %i in (4.0, 2.0, 1.0, 0.5, 0.25, 0.125) DO ^
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr %i ^
--epochs 250 ^
--input-size 3 224 224 ^
--experiment %cexp%-lr-%i-epoch-250 ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml

for %i in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125) DO ^
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr %i ^
--epochs 250 ^
--input-size 3 224 224 ^
--experiment %cexp%-no-pretrain-lr-%i-epoch-250 ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default-no-pretrained.yaml

```

Test also TL from 5 individual dataset DARE models. For DC models, drop heads
```
set cexp=no_empty

for %j in (death-certificates-1, death-certificates-2) DO ^
for %i in (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625) DO ^
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr %i ^
--input-size 3 224 224 ^
--experiment %cexp%-tl-%j-lr-%i ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--initial-checkpoint %EXPDIR%\%j\last.pth.tar ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml ^
--drop-modules classifier*

for %j in (police-register-sheets-1, police-register-sheets-2, swedish-records-birth-dates) DO ^
for %i in (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625) DO ^
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr %i ^
--input-size 3 224 224 ^
--experiment %cexp%-tl-%j-lr-%i ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--initial-checkpoint %EXPDIR%\%j\last.pth.tar ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml

```

**NOTE**: For DC-1, a run with lr == 4.0 started, as the best former was 2.0, which was a border point

### Train
```
set cexp=no_empty

python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 0.25 ^
--input-size 3 224 224 ^
--experiment %cexp%-tl-lr-0.25-epoch-250 ^
--output %EXPDIR%\atlass ^
--formatter dates_ddmyy ^
--initial-checkpoint %EXPDIR%\split-ddmyy\last.pth.tar ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml

python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 2.0 ^
--epochs 250 ^
--input-size 3 224 224 ^
--experiment %cexp%-lr-2.0-epoch-250 ^
--output %EXPDIR%\atlass ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml

python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 0.5 ^
--epochs 1000 ^
--input-size 3 224 224 ^
--experiment %cexp%-no-pretrain-lr-0.5-epoch-1000 ^
--output %EXPDIR%\atlass ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default-no-pretrained.yaml

```

Test also TL from 5 individual dataset DARE models. For DC models, drop heads
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 2.0 ^
--input-size 3 224 224 ^
--experiment %cexp%-tl-death-certificates-1-lr-2.0 ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--initial-checkpoint %EXPDIR%\death-certificates-1\last.pth.tar ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml ^
--drop-modules classifier*

python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 0.25 ^
--input-size 3 224 224 ^
--experiment %cexp%-tl-death-certificates-2-lr-0.25 ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--initial-checkpoint %EXPDIR%\death-certificates-2\last.pth.tar ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml ^
--drop-modules classifier*

python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 0.125 ^
--input-size 3 224 224 ^
--experiment %cexp%-tl-police-register-sheets-1-lr-0.125 ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--initial-checkpoint %EXPDIR%\police-register-sheets-1\last.pth.tar ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml

python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 0.25 ^
--input-size 3 224 224 ^
--experiment %cexp%-tl-police-register-sheets-2-lr-0.25 ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--initial-checkpoint %EXPDIR%\police-register-sheets-2\last.pth.tar ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml

python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--lr 1.0 ^
--input-size 3 224 224 ^
--experiment %cexp%-tl-swedish-records-birth-dates-lr-1.0 ^
--output %EXPDIR%\atlass\lr-search ^
--formatter dates_ddmyy ^
--initial-checkpoint %EXPDIR%\swedish-records-birth-dates\last.pth.tar ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml

```

### Evaluate
```
set cexp=no_empty
set models=(tl-lr-0.25-epoch-250, no-pretrain-lr-0.5-epoch-1000, lr-2.0-epoch-250, tl-death-certificates-1-lr-2.0, tl-death-certificates-2-lr-0.25, tl-police-register-sheets-1-lr-0.125, tl-police-register-sheets-2-lr-0.25, tl-swedish-records-birth-dates-lr-1.0)

for %i in %models% DO ^
python evaluate.py ^
--input-size 3 224 224 ^
--output %EVALDIR%\atlass\%cexp%-%i ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3 ^
--checkpoint %EXPDIR%\atlass\%cexp%-%i\last.pth.tar

```
