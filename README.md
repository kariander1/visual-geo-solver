# Visual Diffusion Models are Geometric Solvers
## Description
Official implementation of "Visual Diffusion Models are Geometric Solvers" paper.

<a href="https://arxiv.org/abs/2510.21697"><img src="https://img.shields.io/badge/arXiv-GeometricSolvers-b31b1b.svg" height=20.5></a>


### 1. **System Dependencies**
Install required build tools:
```bash
sudo apt-get update
sudo apt-get install build-essential libtool libtool-bin autotools-dev automake
```

### 2. **Python Environment**
Create a virtual environment and install dependencies with `uv`:
```
uv venv
uv sync
```
Activate the virtual environment:
```bash
source .venv/bin/activate
```

### 3. Training models
Run `./train.sh` providing the number of gpus to use and the config file name (without the .yaml extension):
```bash
./train.sh <num_gpus> <config_file>
```

### 4. Datasets and model checkpoints
The training and evaluation datasets used in the paper are available at https://huggingface.co/datasets/nirgoren/geometric-solver/tree/main. Download and extract the dataset under the `data` directory.

Trained model checkpoints from the paper are available at https://huggingface.co/nirgoren/geometric-solver/tree/main. Download and extract the checkpoints under the `model` directory.


### 5. Testing models

Run the evaluations scripts under the `eval_scripts` folder.

## Citation
If you use this code for your research, please cite the following work: 
```
@misc{goren2025visualdiffusionmodelsgeometric,
      title={Visual Diffusion Models are Geometric Solvers}, 
      author={Nir Goren and Shai Yehezkel and Omer Dahary and Andrey Voynov and Or Patashnik and Daniel Cohen-Or},
      year={2025},
      eprint={2510.21697},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.21697}, 
}
