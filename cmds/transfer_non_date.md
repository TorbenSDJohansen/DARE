## Variables
For ease of use, define variables that are used in commands below:
```
set DATADIR=Z:\data_cropouts\Labels\DARE
set EXPDIR=Z:\faellesmappe\tsdj\DARE\tl-non-dates
set EVALDIR=Z:\faellesmappe\tsdj\DARE\eval-tl-non-dates

set EVAL_BATCHSIZE=2048
```

# Prepare data from ENS workspace

```
python workspace_to_label.py --wsp-dir Z:\data_cropouts\Labels\DARE\swedish-grades\fromDaniel --output-dir Z:\data_cropouts\Labels\DARE\swedish-grades\labels --share-test 0.7557 --add-filename-hash-to-seed
```

# Train

From DARE (from `full-ddmyyyy`)
```
python train.py ^
--lr 0.25 ^
--input-size 3 224 224 ^
--experiment tl-lr-0.25-epoch-250 ^
--output %EXPDIR% ^
--formatter grades ^
--initial-checkpoint Z:\faellesmappe\tsdj\DARE\experiments\full-ddmyyyy\last.pth.tar --drop-modules classifier* ^
--dataset swedish-grades ^
--data_dir %DATADIR% ^
--config Z:\faellesmappe\tsdj\DARE\experiments\cfgs\default.yaml ^
--initial-log ^
--log-wandb
```

From ImageNet21k
```
python train.py ^
--lr 2.0 ^
--epochs 250 ^
--input-size 3 224 224 ^
--experiment lr-2.0-epoch-250 ^
--output %EXPDIR% ^
--formatter grades ^
--dataset swedish-grades ^
--data_dir %DATADIR% ^
--config Z:\faellesmappe\tsdj\DARE\experiments\cfgs\default.yaml ^
--initial-log ^
--log-wandb
```

Without any pre-training
```
python train.py ^
--lr 0.5 ^
--epochs 1000 ^
--input-size 3 224 224 ^
--experiment no-pretrain-lr-0.5-epoch-1000 ^
--output %EXPDIR% ^
--formatter grades ^
--dataset swedish-grades ^
--data_dir %DATADIR% ^
--config Z:\faellesmappe\tsdj\DARE\experiments\cfgs\default-no-pretrained.yaml ^
--initial-log ^
--log-wandb
```

# Evaluate

From DARE (from `full-ddmyyyy`)
```
python evaluate.py ^
--lr 0.25 ^
--input-size 3 224 224 ^
--output %EVALDIR%/tl-lr-0.25-epoch-250 ^
--formatter grades ^
--dataset swedish-grades ^
--data_dir %DATADIR% ^
--config Z:\faellesmappe\tsdj\DARE\experiments\cfgs\default.yaml ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3 ^
--checkpoint %EXPDIR%\tl-lr-0.25-epoch-250\last.pth.tar
```

From ImageNet21k
```
python evaluate.py ^
--lr 0.25 ^
--input-size 3 224 224 ^
--output %EVALDIR%/lr-2.0-epoch-250 ^
--formatter grades ^
--dataset swedish-grades ^
--data_dir %DATADIR% ^
--config Z:\faellesmappe\tsdj\DARE\experiments\cfgs\default.yaml ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3 ^
--checkpoint %EXPDIR%\lr-2.0-epoch-250\last.pth.tar
```

Without any pre-training
```
python evaluate.py ^
--lr 0.25 ^
--input-size 3 224 224 ^
--output %EVALDIR%/no-pretrain-lr-0.5-epoch-1000 ^
--formatter grades ^
--dataset swedish-grades ^
--data_dir %DATADIR% ^
--config Z:\faellesmappe\tsdj\DARE\experiments\cfgs\default.yaml ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3 ^
--checkpoint %EXPDIR%\no-pretrain-lr-0.5-epoch-1000\last.pth.tar
```