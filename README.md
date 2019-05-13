# Feature_Critic_VD
Demo code for 'Feature-Critic Networks for Heterogeneous Domain Generalization'
. The paper is located at https://arxiv.org/abs/1901.11448.

>  Yiying Li, Yongxin Yang, Wei Zhou, Timothy M. Hospedales. Feature-Critic Networks for Heterogeneous Domain Generalization[C]. ICML 2019.

### Dataset
The example code mainly reproduces the experimental results of Heterogeneous DG experiments with VD,
so it is necessary to download the corresponding data set from the official website(https://www.robots.ox.ac.uk/%7Evgg/decathlon/).
please download the following files:
```
(1) Annotations and code. The devkit [22MB] contains the annotation files as well as example MATLAB code for evaluation (using this code is not a requirement).
(2) Images. The following archives contain the preprocessed images for each dataset:
            Preprocessed images [406MB]. Images from all datasets except ImageNet ILSVRC.
            Preprocessed ILSVRC images [6.1GB]. In order to download the data, the attendees are required to register an ImageNet (http://image-net.org/signup) account first. Images for the ImageNet ILSVRC dataset (this is shipped separately due to copyright issues).
```

### Installation

Install Anaconda:
```
curl -o /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash /tmp/miniconda.sh
conda create -n FC_VD python=2.7.12
source activate FC_VD
```
Install necessary Python packages:
```
pip install torchvision pycocotools torch
```

### Running
First go to the Feature_Critic_VD code folder:
```
cd <path_to_Feature_Critic_VD_folder>
```
Then launch the entry script of baseline method:
```
python main_baseline.py
```
Experiment data is saved in `<home_dir>/logs`.

Run the Feature_Critic_VD:
```
python main_Feature_Critic.py
```
### Bibtex
 ```
 @inproceedings{Li2019ICML,
   title={Feature-Critic Networks for Heterogeneous Domain Generalization},
   author={Li, Yiying and Yang, Yongxin and Zhou, Wei and Hospedales, Timothy},
  	booktitle={The Thirty-sixth International Conference on Machine Learning},
  	year={2019}
 }
 ```
 ### Your own data
 Please tune the folder <VD> for your own data.