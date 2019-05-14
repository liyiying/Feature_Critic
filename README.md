# Feature_Critic
Demo code for 'Feature-Critic Networks for Heterogeneous Domain Generalisation', including codes for heterogeneous DG (VD) and homogeneous DG (PACS).
This paper is located at https://arxiv.org/abs/1901.11448 and will appear in the forthcoming ICML 2019.

>  Yiying Li, Yongxin Yang, Wei Zhou, Timothy M. Hospedales. Feature-Critic Networks for Heterogeneous Domain Generalisation[C]. ICML 2019.

## Introduction
Feature_Critic aims to address the domain generalisation problem with a particular focus on the heterogeneous case, by meta-learning a regulariser to help train a feature extractor to be domain invariant. The resulting feature extractor outperforms alternatives for general purpose use as a fixed downstream image encoding. Evaluated on Visual Decathlon -- the largest DG evaluation thus far -- this suggests that Feature_Critic trained feature extractors could be of wide potential value in diverse applications. Furthermore Feature_Critic also performs favourably compared to state-of-the-art in the homogeneous DG setting, such as on PACS dataset.

## Citing Feature_Critic
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


All trained models of VD and PACS are on  https://mega.nz/#F!rRkgzawL!qoGX4bT3sif88Ho1Ke8j1Q, and they can be put under `<home_dir>/model_output/`. **If you want to use the trained Feature_Critic for encoding to extract your features, you can download and use the torch models under ```<Feature_Critic>``` folder.**

VD dataset download should follow the Download VD Dataset instructions below.

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
Make sure to run script to download the PACS dataset and trained models from the MEGA network disk.
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

## Running and Results

### Experiments on VD
1. Baseline(AGG)
Launch the entry script of baseline method:
```
python main_baseline.py --dataset=VD
```
Parameters (e.g., learning_rate, batch_size...) and flags can be found and tuned in `main_baseline.py`. Turn on the `is_train` to train the baseline model.
Experiment data is saved in `<home_dir>/logs/VD/baseline/`. You can achieve 19.56%, 36.49%, 58.04%, 46.98% on the four target domains (Aircraft, D.Textures, VGG-Flowers, UCF101) with the average 40.27%. (cf Table 1 in the paper)

2. Feature_Critic
Load the VD baseline model to `<home_dir>/model_output/VD/baseline/`

Launch the entry script of Feature_Critic method, parameters and flags can also be tuned by yourself:
```
python main_Feature_Critic.py --dataset=VD
```
Experiment data is saved in `<home_dir>/logs/VD/Feature_Critic/`. You can achieve 20.94%, 38.88%, 58.53%, 50.82% on the four target domains (Aircraft, D.Textures, VGG-Flowers, UCF101) with the average 42.29%. (cf Table 1 in the paper)


### Experiments on PACS
Experiments need to be performed four times in the leave-one-domain-out way. Take the "leave-A-domain-out" as the example, and you can change the target domain (`unseen_index`) as in the main file.

For baseline, you can achieve 63.3%, 66.3%, 88.6%, 56.5% when setting A, C, P, S as the target domain，respectively, and get the average 68.7%. (cf Table 5 in the paper)

For Feature_Critic, you can achieve 64.4%, 68.6%, 90.1%, 58.4% when setting A, C, P, S as the target domain，respectively, and get the average 70.4%. (cf Table 5 in the paper)

1. Baseline(AGG)
Launch the entry script of baseline method:
```
python main_baseline.py --dataset=PACS
```
Parameters (e.g., learning_rate, batch_size...) and flags can be found and tuned in `main_baseline.py`. Turn on the `is_train` to train the baseline model.
Experiment data is saved in `<home_dir>/logs/PACS/baseline/A/`. 

2. Feature_Critic
Load the PACS baseline model (A) to `<home_dir>/model_output/PACS/baseline/A/`

Launch the entry script of Feature_Critic method, parameters and flags can also be tuned by yourself:
```
python main_Feature_Critic.py --dataset=PACS
```
Experiment data is saved in `<home_dir>/logs/PACS/Feature_Critic/A/`.


