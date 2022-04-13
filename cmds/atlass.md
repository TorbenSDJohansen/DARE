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

**TODO**: Tune epochs
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
--input-size 3 224 224 ^
--experiment %cexp%-no-pretrain-lr-%i ^
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
--lr 4.0 ^
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
```
set cexp=no_empty

python evaluate.py ^
--input-size 3 224 224 ^
--output %EVALDIR%\atlass\%cexp%-tl ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3 ^
--checkpoint %EXPDIR%\atlass\%cexp%-tl\last.pth.tar

python evaluate.py ^
--input-size 3 224 224 ^
--output %EVALDIR%\atlass\%cexp% ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3 ^
--checkpoint %EXPDIR%\atlass\%cexp%\last.pth.tar

python evaluate.py ^
--input-size 3 224 224 ^
--output %EVALDIR%\atlass\%cexp%-no-pretrain ^
--formatter dates_ddmyy ^
--dataset atlass ^
--data_dir %DATADIR% ^
--labels-subdir %cexp%  ^
--config %EXPDIR%\cfgs\default.yaml ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3 ^
--checkpoint %EXPDIR%\atlass\%cexp%-no-pretrain\last.pth.tar

```

