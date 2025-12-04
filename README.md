# DARE
Code related to the paper [DARE: A large-scale handwritten DAte REcognition system](https://arxiv.org/abs/2210.00503) by Christian M. Dahl, Torben Johansen, Emil N. Sørensen, Christian Westermann, and Simon Wittrock.

- [Download Database](#download-database)
- [Clone Repository and Prepare Environment](#clone-repository-and-prepare-environment)
- [Replicate Results](#replicate-results)
- [License](#license)
- [Citing](#citing)

## Download Database
The DARE database can be downloaded from [Zenodo](https://zenodo.org/records/17589563).
It consists of seven datasets, each with its own train and test split.
The number of observation in each dataset varies beteen ~10k and ~1M.

## Clone Repository and Prepare Environment
To get started, first clone the repository locally:
```
git clone https://github.com/TorbenSDJohansen/DARE
```

Then prepare an environment (here using conda and the name `DARE`):
```
conda create -n DARE numpy pandas pillow scikit-learn tensorboard opencv matplotlib pyyaml
conda activate DARE
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install imutils timm=0.6.7
```

After making sure all dependencies are installed, use the following code to install `timmsn`.
```
pip install path/to/timm-sequence-net
```
**Note**: `timmsn=0.2.5` used.

### Model Zoo
Our trained models are available for download:

<details>

<summary>
Single-dataset models
</summary>

| model    | Sequence          | SeqAcc        | url                                                                             |
| ---      | ---               | ---           | ---                                                                             |
| M-DC-1   | DD-M-YY(YY)       | 97.5 (97.3)   | [model](https://www.dropbox.com/scl/fi/91s1bzgiaculw9f6on5pq/death-certificates-1.pth.tar?rlkey=9cvkx6d0itjzyf9kl7vr0jxo3&dl=0)                                                                             |
| M-DC-2   | DD-M-YY(YY)       | 92.9 (92.8)   | [model](https://www.dropbox.com/scl/fi/ulpb3ecu19jfghpnjva23/death-certificates-2.pth.tar?rlkey=9p2ma697uwnlpt3ex8o65o4in&dl=0)                                                                             |
| M-PR-1   | DD-M-YY           | 93.5          | [model](https://www.dropbox.com/scl/fi/4ykh85ly8v9irxziux39h/police-register-sheets-1.pth.tar?rlkey=txu55i04n2uz7ewaoxsnrmybo&dl=0)                                                                             |
| M-PR-2   | DD-M-YY           | 84.2          | [model](https://www.dropbox.com/scl/fi/ud8d3qgth342vf0q8ej0c/police-register-sheets-2.pth.tar?rlkey=m2oj9z69n4vbs4iiaj45807fu&dl=0)                                                                             |
| M-SWE-BD | DD-M-YY           | 94.5          | [model](https://www.dropbox.com/scl/fi/3dimj1h8uoppz59fqgbmc/swedish-records-birth-dates.pth.tar?rlkey=hdnlfe04lktxs8v6rxqnxh71f&dl=0)                                                                             |
| M-FR     | DD-M              | 98.2          | [model](https://www.dropbox.com/scl/fi/aag22ovnfosxgn8rsi2wd/funeral-records.pth.tar?rlkey=30jr07d0ocanxz0jtnmk07pxx&dl=0)                                                                             |
| M-NHVD   | DD-M              | 97.5          | [model](https://www.dropbox.com/scl/fi/yjlvuvf5ziu52tqz9nax4/cihvr.pth.tar?rlkey=5srdidk01p2egklbklqfhots5&dl=0)                                                                             |
| M-SWE-DD | DD-M              | 99.4          | [model](https://www.dropbox.com/scl/fi/8m616t6zt7f1ehdm8cpny/swedish-records-death-dates.pth.tar?rlkey=09ydt5bj409hfb5xpy6xb9bgx&dl=0)                                                                             |

</details>

<details>

<summary>
Multi-dataset models
</summary>

| model     | Sequence  | SeqAcc @ DC-1 | SeqAcc @ DC-2 | SeqAcc @ PR-1 | SeqAcc @ PR-2 | SeqAcc @ SBD | SeqAcc @ FR | SeqAcc @ NHVD | SeqAcc @ SDD | url |
| ---       | ---       | ---           | ---           | ---           | ---           | ---             | ---         | ---           | ---             | --- |
| M-DDMYYYY | DD-M-YY   | 97.7          | 91.7          | 94.0          | 85.2          | 94.7            | 98.2        | 97.5          | 99.3            | [model](https://www.dropbox.com/scl/fi/0byqf68umo1par76wgjn9/full-ddmyyyy.pth.tar?rlkey=p6iclol7wq2nqzq7ft808z0uf&dl=0) |
| M-DDMYY   | DD-M-YY   | 97.6          | 91.9          | 94.0          | 85.2          | 94.7            |              |               |                 | [model](https://www.dropbox.com/scl/fi/b0v990y1yzuxa7gi76klq/split-ddmyy.pth.tar?rlkey=9whwo6oihpwmt98r5f5f5sj9p&dl=0) |
| M-DDM     | DD-M      |               |               |               |               |                 | 98.2         | 97.5          | 99.4            | [model](https://www.dropbox.com/scl/fi/ikyc6ywchbhomm6qdb5bj/split-ddm.pth.tar?rlkey=bi2cbbm3bzbkloo40z1j6fob0&dl=0) |

</details>

</details>

<details>

<summary>
Swedish gradesheets transfer learning models
</summary>

| Pretrained  | Sequence | SecAcc | TL Gain  | Error Rate Reduction | url       |
| ---         | ---      | ---    | ---      | ---                  | ---       |
| No          | DD-M-YY  | 85.2   |          |                      | [model](https://www.dropbox.com/scl/fi/fj78bmrb1dpcl7833xh81/atlass-no-pretrain.pth.tar?rlkey=d3a2m5nlkz6ki3xyax13oml9s&dl=0) |
| ImageNet21k | DD-M-YY  | 93.4   | + 8.2    | - 55.4               | [model](https://www.dropbox.com/scl/fi/2lcxk8nbl007xhdp75ksr/atlass-IN21k.pth.tar?rlkey=u6kr0fa7q2mqfcj3x2dampslu&dl=0) |
| M-DC-1      | DD-M-YY  | 95.2   | + 10.0   | - 67.6               | [model](https://www.dropbox.com/scl/fi/zk9422ufkzaase9gxz2bq/atlass-death-certificates-1.pth.tar?rlkey=ogs0pl7fa0gdnau4izk33ma8h&dl=0) |
| M-PR-2      | DD-M-YY  | 95.4   | + 10.2   | - 68.9               | [model](https://www.dropbox.com/scl/fi/23umr7hkve8d3l4pw0kny/atlass-police-register-sheets-2.pth.tar?rlkey=69zwf92huowvikui97gjwn608&dl=0) |
| M-DC-2      | DD-M-YY  | 95.6   | + 10.4   | - 70.5               | [model](https://www.dropbox.com/scl/fi/1848ha39adzfl2cve90xp/atlass-death-certificates-2.pth.tar?rlkey=2uf2o5ka8vkgrl0abx8qobs9q&dl=0) |
| M-SWE-BD    | DD-M-YY  | 95.6   | + 10.4   | - 70.5               | [model](https://www.dropbox.com/scl/fi/10b1ksatrws71yx1nhxtv/atlass-swedish-records-birth-dates.pth.tar?rlkey=c05pi1gajtgde4u6vu8tkaf14&dl=0) |
| M-PR-1      | DD-M-YY  | 95.9   | + 10.7   | - 72.3               | [model](https://www.dropbox.com/scl/fi/ka17welpo1yijva6wsvjz/atlass-police-register-sheets-1.pth.tar?rlkey=aax6pj80tem47pwp6uhzjrmfk&dl=0) |
| M-DDMYY     | DD-M-YY  | 96.2   | + 11.0   | - 74.3               | [model](https://www.dropbox.com/scl/fi/ks3fikgh6fojqvatrj1nu/atlass-ddmyy.pth.tar?rlkey=nb2h9xxuvsj6qkz9fy87eyctw&dl=0) |

</details>


## Replicate Results
You can find code to replicate all results in [cmds/cmds.md](cmds/cmds.md) (main results) and [cmds/atlass.md](cmds/atlass.md) (transfer learning).
Note that you have to change the folder names at the top of the documents to match your local paths, and that the commands are tested on Windows Command Prompt and slight modifications might be needed depending on the console.

For our ablation results, see [cmds/ablation.md](cmds/ablation.md).

For our transfer learning to non-date dataset results, see [cmds/transfer_non_date.md](cmds/transfer_non_date.md).

For our experiments with synthetic data, see [cmds/synthetic.md](cmds/synthetic.md).

## License
Our code is licensed under MIT (see [LICENSE](LICENSE)).

## Citing
If you would like to cite our work, please use:
```bibtex
@article{dahl2022dare,
  title={DARE: A large-scale handwritten {DA}te {RE}cognition system},
  author={Dahl, Christian M. and Johansen, Torben S. D. and S{\o}rensen, Emil N. and Westermann, Christian E. and Wittrock, Simon F.},
  journal={arXiv preprint arXiv:2210.00503},
  year={2022}
}
```
