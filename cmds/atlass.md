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

### Train
**TODO**: Fix LR, epochs
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

### Evaluate
```
set cexp=no_empty
set models=(tl-lr-0.25-epoch-250, no-pretrain-lr-0.5-epoch-1000, lr-2.0-epoch-250)

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

