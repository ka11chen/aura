# AURA Setup

## Python venv
(Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# (.venv) >
pip install -r requirements.txt
```

## Autogen setup
- enter your api keys in .env file

## Instruct-pix2pix setup
> The setup is for generating modified skeletons. 
This requires CUDA; however, suggestion generation is still functional without this.

- install miniconda3
```
git clone https://github.com/timothybrooks/instruct-pix2pix.git

cd instruct-pix2pix

conda env create -f environment.yaml
conda activate ip2p
```

- download http://instruct-pix2pix.eecs.berkeley.edu/instruct-pix2pix-00-22000.ckpt

- put it to ./checkpoints/

```
pip install --upgrade transformers
```

- check CUDA version

```
nvidia-smi
```

- install PyTorch CUDA，pytorch-cuda <= driver CUDA Version

```
pip uninstall torch torchvision torchaudio

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Execute
```bash
cd web
python app.py
```
Then visit http://localhost:5000
> Sometimes this port is in used. If that happened change `app.run(..., port=5000)` in the end of app.py to other random port.