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
python workspace_to_label.py --wsp-dir Z:\data_cropouts\Labels\DARE\swedish-grades\fromDaniel --output-dir Z:\data_cropouts\Labels\DARE\swedish-grades\labels --share-test 0.5 --add-filename-hash-to-seed
```

# Train

From ImageNet21k
```
python train.py --formatter grades --output %EXPDIR% --dataset swedish-grades --experiment swedish-grades --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --initial-log --log-wandb
```

From DARE (from `full-ddmyyyy`)
```
python train.py --formatter grades --output %EXPDIR% --dataset swedish-grades --experiment swedish-grades-tl --config cfgs/efficientnetv2_s.yaml --data_dir %DATADIR% --initial-log --initial-checkpoint Z:\faellesmappe\tsdj\DARE\experiments\full-ddmyyyy\last.pth.tar --drop-modules classifier* --log-wandb
```

# Evaluate
