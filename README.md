# geometric-solver

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

### 4. Testing models
Run the evaluations scripts under the `eval_scripts` folder.