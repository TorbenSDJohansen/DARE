# DARE
Code related to the paper DARE: A large-scale handwritten DAte REcognition system by Christian M. Dahl, Torben Johansen, Emil N. Sørensen, Christian Westermann, and Simon Wittrock.

- [Download Database](#download-database)
- [Clone Repository and Prepare Environment](#clone-repository-and-prepare-environment)
- [Replicate Results](#replicate-results)
- [License](#license)
- [Citing](#citing)
- [TODO](#todo)

## Download Database

## Clone Repository and Prepare Environment
To get started, first clone the repository locally:
```
git clone https://github.com/TorbenSDJohansen/DARE
```

Then prepare an environment (here using conda and the name `timmsn`):
```
conda create -n DARE numpy pandas pillow scikit-learn tensorboard opencv matplotlib
conda activate DARE
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
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
@article{tsdj2022dare,
  author = {XXX},
  title = {XXX},
  year = {XXX},
  journal = {XXX},
}
```

## TODO