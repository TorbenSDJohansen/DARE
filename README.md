# DARE
Code related to the paper [DARE: A large-scale handwritten DAte REcognition system](https://arxiv.org/abs/2210.00503) by Christian M. Dahl, Torben Johansen, Emil N. Sørensen, Christian Westermann, and Simon Wittrock.

- [Download Database](#download-database)
- [Clone Repository and Prepare Environment](#clone-repository-and-prepare-environment)
- [Replicate Results](#replicate-results)
- [License](#license)
- [Citing](#citing)

## Download Database
The DARE database can be downloaded from [Kaggle](https://www.kaggle.com/datasets/sdusimonwittrock/dare-database).
It consists of seven datasets, each with its own train and test split.
The number of observation in each dataset varies beteen ~10k and ~1M.

## Clone Repository and Prepare Environment
To get started, first clone the repository locally:
```
git clone https://github.com/TorbenSDJohansen/DARE
```

Then prepare an environment (here using conda and the name `timmsn`):
```
conda create -n DARE numpy pandas pillow scikit-learn tensorboard opencv matplotlib pyyaml
conda activate DARE
conda install pytorch=1.9 torchvision=0.10 torchaudio cudatoolkit=10.2 -c pytorch
pip install imutils timm=0.5.4
```

After making sure all dependencies are installed, use the following code to install `timmsn`.
```
pip install path/to/timm-sequence-net
```

### Model Zoo

## Replicate Results

## License

Our code is licensed under MIT (see [LICENSE](LICENSE)).

## Citing
If you would like to cite our work, please use:
```bibtex
@article{dahl2022dare,
  title={DARE: A large-scale handwritten {DA}te {Re}cognition system},
  author={Dahl, Christian M. and Johansen, Torben S. D. and S{\o}rensen, Emil N. and Westermann, Christian E. and Wittrock, Simon F.},
  journal={arXiv preprint arXiv:2210.00503},
  year={2022}
}
```
