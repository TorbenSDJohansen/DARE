# DARE
Code related to the paper [DARE: A large-scale handwritten DAte REcognition system](https://arxiv.org/abs/2210.00503) by Christian M. Dahl, Torben Johansen, Emil N. Sørensen, Christian Westermann, and Simon Wittrock.

- [Download Database](#download-database)
- [Clone Repository and Prepare Environment](#clone-repository-and-prepare-environment)
- [Replicate Results](#replicate-results)
- [License](#license)
- [Citing](#citing)

## Download Database

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

## Replicate Results
You can find code to replicate all results in [cmds/cmds.md](cmds/cmds.md) and [cmds/atlass.md](cmds/atlass.md).

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
