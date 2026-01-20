# AURA Setup

## Python venv
(Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# (.venv) >
pip install -r requirements.txt
```

## autogen setup
- enter your api keys in .env file

## instruct-pix2pix

install miniconda3

```
git clone https://github.com/timothybrooks/instruct-pix2pix.git

cd instruct-pix2pix

conda env create -f environment.yaml
conda activate ip2p
```

download http://instruct-pix2pix.eecs.berkeley.edu/instruct-pix2pix-00-22000.ckpt

put it to ./checkpoints/

```
pip install --upgrade transformers
```

check CUDA version

```
nvidia-smi
```

install PyTorch CUDA，pytorch-cuda <= driver CUDA Version

```
pip uninstall torch torchvision torchaudio

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
