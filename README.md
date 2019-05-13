# Feature_Critic
Demo code for 'Feature-Critic Networks for Heterogeneous Domain Generalisation', including codes for heterogeneous DG (VD) and homogeneous DG (PACS).
This paper is located at https://arxiv.org/abs/1901.11448 and will appear in the forthcoming ICML 2019.

>  Yiying Li, Yongxin Yang, Wei Zhou, Timothy M. Hospedales. Feature-Critic Networks for Heterogeneous Domain Generalisation[C]. ICML 2019.

# Introduction
Feature_Critic aims to address the domain generalisation problem with a particular focus on the heterogeneous case, by meta-learning a regulariser to help train a feature extractor to be domain invariant. The resulting feature extractor outperforms alternatives for general purpose use as a fixed downstream image encoding. Evaluated on Visual Decathlon -- the largest DG evaluation thus far -- this suggests that Feature_Critic trained feature extractors could be of wide potential value in diverse applications. Furthermore Feature_Critic also performs favourably compared to state-of-the-art in the homogeneous DG setting, such as on PACS dataset.

# Citing Feature_Critic
If you find Feature_Critic useful in your research, please consider citing:
 ```
 @inproceedings{Li2019ICML,
    Author={Li, Yiying and Yang, Yongxin and Zhou, Wei and Hospedales, Timothy},
    Title={Feature-Critic Networks for Heterogeneous Domain Generalisation},
    Booktitle={The Thirty-sixth International Conference on Machine Learning},
    Year={2019}
    }
 ```
 
 ## Download datasets and models
 
 ### Preparation
We provide two ways to download datasets and trained models on our MEGA network disk:
(i) Download directly from the link and put them under the corresponding project dir:
PACS dataset is on  https://mega.nz/#F!jBllFAaI!gOXRx97YHx-zorH5wvS6uw. pacs_data and pacs_label can be put under ```<home_dir>/data/PACS/```.
All trained models of VD and PACS are on  https://mega.nz/#F!rRkgzawL!qoGX4bT3sif88Ho1Ke8j1Q, and they can be put under `<home_dir>/model_output/`.  ＊＊The models under the ```<Feature_Critic>``` folder are the Feature_Critic trained models that can be utilized directly.＊＊
VD dataset download should follow the instructions below.

(ii) Install the network disk command line tool first and then use our script for downloading.
```
(1) Download the soure code of MEGA tool.
git clone https://github.com/meganz/MEGAcmd.git
cd MEGAcmd
git submodule update --init --recursive
(2) Install the tool
apt install libcurl4-openssl-dev libc-ares-dev libssl-dev libcrypto++-dev zlib1g-dev libsqlite3-dev libfreeimage-dev
apt install autoconf automake libtool libreadline6-dev
sh autogen.sh
./configure
make
sudo make install
sudo ldconfig
```

### Download VD Dataset
From the official website(https://www.robots.ox.ac.uk/%7Evgg/decathlon/), please download the following files:
```
(1) Annotations and code. The devkit [22MB] contains the annotation files as well as example MATLAB code for evaluation. You can put under `<home_dir>/data/VD/`.
(2) Images. The following archives contain the preprocessed images for each dataset, and they can be put under `<home_dir>/data/`:
            Preprocessed images [406MB]. Images from all datasets except ImageNet ILSVRC.
            Preprocessed ILSVRC images [6.1GB]. In order to download the data, the attendees are required to register an ImageNet (http://image-net.org/signup) account first. Images for the ImageNet ILSVRC dataset (this is shipped separately due to copyright issues).
```

### Download PACS Dataset and trained models
Make sure to run script to download the PACS dataset and trained models from the MEGA netowrk disk.
 ```
 bash get_model_dataset.sh
 ```

## Installation

Install Anaconda:
```
curl -o /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash /tmp/miniconda.sh
conda create -n FC python=2.7.12
source activate FC
```
Install necessary Python packages:
```
pip install torchvision pycocotools torch
```

## Running

### Experiments on VD
1. Baseline(AGG)
Launch the entry script of baseline method:
```
python main_baseline.py --dataset=VD
```
Parameters (e.g., learning_rate, batch_size) and flags can be found and tuned in `main_baseline.py`. Turn on the `is_train` to train the baseline model.
Experiment data is saved in `<home_dir>/logs`.

2. Feature_Critic
Load the VD baseline model under the ....

Launch the entry script of Feature_Critic method, parameters and flags can also be tuned by yourself:
```
python main_Feature_Critic.py --dataset=VD
```

### Experiments on PACS


