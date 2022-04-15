# Context-Aware Sequence Alignment using 4D Skeletal Augmentation
Taein Kwon, Bugra Tekin, Sigyu Tang, and Marc Pollefeys
<img src="baseball_swing.gif" alt="CASA" width="750"/>

This code is for Context-Aware Sequence Alignment using 4D Skeletal Augmentation, CVPR 2022 (oral). You can see more details about more paper in our [project page](https://taeinkwon.com/projects/casa/). Note that we referred [LoFTR](https://github.com/zju3dv/LoFTR) to implement our framework.

## Environment Setup
To setup the env, 
```
git clone https://github.com/taeinkwon/casa_clean.git
cd CASA
conda env create -f env.yml
conda activate CASA
```

## External Dependencies

### Folder Structures
<pre>
.
├── bodymocap
├── extra_data
│   └── body_module
│         └── J_regressor_extra_smplx.npy
├── human_body_prior
│   └── ...
├── manopth
│   ├── __init__.py
│   ├── arguitls.py
│   └── ...
├── smpl
│   └── models
│         └── SMPLX_NEUTRAL.pkl
├── mano 
│   ├── models
│   │     ├── MANO_LEFT.pkl
│   │     └── MANO_RIGHT.pkl
│   ├── websuers
│   ├── __init__.py
│   └── LICENSE.txt
├── npyrecords
├── sripts
├── src
└── ...
</pre>

### Install MANO and Manopth
In this repository, we use the [MANO](https://mano.is.tue.mpg.de/) model from MPI and some part of [Yana](https://hassony2.github.io/)'s code for hand pose alignment.
- Clone [manopth](https://github.com/hassony2/manopth) ```git clone https://github.com/hassony2/manopth.git``` and copy ```manopth``` and ```mano``` folder (inside) into the CASA folder.
- Go to the [mano website](https://mano.is.tue.mpg.de/) and download models and code and put them in ```CASA/mano/models```.
- In ```smpl_handpca_wrapper_HAND_only.py```, please change following lines to run in python3. L23:import CPickle as pickle -> import pickle, L30: dd = pickle.load(open(fname_or_dick)) -> dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1'), L144: print 'FINTO' -> print('FINTO').

### Install Vposer
We used Vposer to augment body pose.
- In the [Vposer](https://github.com/nghorbani/human_body_prior) repository, clone it ```git clone https://github.com/nghorbani/human_body_prior.git```
- Copy the human_body_prior folder into the CASA folder.
- Go into the human_body_prior folder and run the setup.py
  ```
  cd human_body_prior
  python setup.py develop
  ```

### Install SMPL
We use the SMPL model for body pose alignment.
- Download [SMPL-X](https://smpl-x.is.tue.mpg.de/) ver 1.1. and VPoser V2.0. 
- Put the ```SMPLX_NUETRAL.pkl``` into the ```CASA/smpl/models``` folder. 
- Copy VPoser files in ```CASA/human_body_prior/support_data/downlaods/vposer_v2_05/```
- In order to use the joints from FrankMocap, you need to additionally clone [FrankMocap](https://github.com/facebookresearch/frankmocap) ```git clone https://github.com/facebookresearch/frankmocap.git``` and put the ```bodymocap``` fodler into ```CASA``` fodler. 
- After then, run ```sh download_extra_data.sh``` to get the J_regressor_extra_smplx.npy file.

## Datasets
You can download npy files [here](https://drive.google.com/file/d/16Kgy8iESC-0YwqELxfE9mWu24Jxzsu1C/view?usp=sharing). In the npy files, normalized joints and labels are included. In order to get the original data, you should go to each dataset websites and download the datasets there.

### H2O
We selected ```pouring milk``` sequences and manually divided into train and test set with the new labels we set. Please go to the [H2O project page](https://taeinkwon.com/projects/h2o/) and download the dataset there.

### PennAction
We estimated 3D joints using [FrankMocap](https://github.com/facebookresearch/frankmocap) for the [Penn Action dataset](https://dreamdragon.github.io/PennAction/). Penn Action has 13 different actions: baseball_pitch, baseball_swing, bench_press, bowling, clean_and_jerk, golf_swing, jumping_jacks, pushups, pullups, situp, squats, tennis_forehand, tennis_serve.

### IkeaASM
We downloaded and used the 3D joints from triangulation of 2D poses in the [IkeaASM dataset](https://ikeaasm.github.io/).

## Train
To train the Penn Action dataset,
```
sh scripts/train/pennaction_train.sh ${dataset_name}
```
For example,
```
sh scripts/train/pennaction_train.sh tennis_serve
```

## Eval
We also provide pre-trained models. To evalate the pre-trained model

```
sh scripts/eval/pennaction_eval.sh ${dataset_name} ${eval_model_path}
```
For example,
```
sh scripts/eval/pennaction_eval.sh tennis_serve logs/tennis_serve/CASA=64/version_0/checkpoints/last.ckpt
```

## License
Note that our code follows the Apache License 2.0. However, external libraries follows their own licenses.