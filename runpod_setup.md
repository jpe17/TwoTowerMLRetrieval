# Runpod setup

### 1. Connect to SSH

	ssh -p PORT -i ~/.ssh/MLX_key root@ipadress
### 2. Install conda

	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

	bash miniconda.sh

	source ~/.bashrc
### 3. Create & activate conda env

	conda env create -f conda_env_ext_gpu.yml

	conda activate two_tower_env