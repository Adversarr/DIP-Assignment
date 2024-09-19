# General Envirionment Setup

I'm using following system & env:

- MacBook Pro, Apple M1 2021
- Anaconda, from homebrew

The virtual environment is created via:

```sh
conda activate base
python -m venv .venv --prompt dip
source .venv/bin/activate
pip install -r XXX/requirements.txt
```