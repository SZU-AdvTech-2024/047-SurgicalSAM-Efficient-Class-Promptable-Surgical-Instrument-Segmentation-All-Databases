## Installation

Following [Segment Anything](https://github.com/facebookresearch/segment-anything), the code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. For SurgicalSAM, `python=3.8`, `pytorch=1.11.0`, and `torchvision=0.12.0` are used.

1. Clone the repository.
      ```
      git clone https://github.com/wenxi-yue/SurgicalSAM.git
      cd SurgicalSAM
      ```

2. Create a virtual environment for SurgicalSAM and and activate the environment.
    ```
    conda create -n surgicalsam python=3.8 -y
    conda activate surgicalsam
    ```
3. Install Pytorch and TorchVision. In our case, we use `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) for installation in your specific condition. 

4. Install other dependencies.
    ```
    pip install -r requirements.txt
    ```

## Data
For EndoVis2018 segmentation annotation provided [here](https://github.com/BCV-Uniandes/ISINet) by [3].
For EndoVis2017 provided [here](https://github.com/ternaus/robot-surgery-segmentation).

In SurgicalSAM, we use the pre-computed SAM features since the image encoder is frozen. We provide the pre-computed SAM features and ground-truth annotations [here](https://unisyd-my.sharepoint.com/:f:/g/personal/wenxi_yue_sydney_edu_au/Et9Nz5d4r2BDkDTakggPMFAB11UNIfbUN3PNz71p1XP0Ug?e=HCYll0). You may use our provided pre-computed SAM features or [generate SAM features from scratch](https://github.com/facebookresearch/segment-anything). 

For inference, please follow the inference instructions below. No further data processing is needed. 

For training, we augment the training data and pre-compute their SAM features before training (offline). Alternatively, you can opt for data augmentation during training (online), which provides greater augmentation diversity. Our training data augmentation is performed as below.
```
cd surgicalSAM/tools/
python data_preprocess.py  --dataset endovis_2018  --n-version 40
python data_preprocess.py  --dataset endovis_2017  --n-version 40
```

## Checkpoints

In SurgicalSAM, `vit_h` is used. 

Please find the checkpoint of SAM in `vit_h` version [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). 

The checkpoint of official trained SurgicalSAM [in the ckp.zip here](https://unisyd-my.sharepoint.com/:u:/g/personal/wenxi_yue_sydney_edu_au/EYUxVQR7XxhDlLWsAL57IqIBbYw2RfCgHCslwGAuoNlCHQ?e=bNgjA0).


##  File Organisation
After downloading the data and model checkpoints and preprocessing the data, the files should be organised as follows.

  ```tree
  SurgicalSAM
      |__assets
      |    ...
      |__data
      |    |__endovis_2018
      |    |       |__train
      |    |       |  |__0
      |    |       |  |  |__binary_annotations
      |    |       |  |  |     ...
      |    |       |  |  |__class_embeddings_h
      |    |       |  |  |     ...
      |    |       |  |  |__images
      |    |       |  |  |     ...
      |    |       |  |  |__sam_features_h
      |    |       |  |       ...
      |    |       |  |__1
      |    |       |  |  ...
      |    |       |  |__2
      |    |       |  |  ...
      |    |       |  |__3
      |    |       |  |  ...
      |    |       |  |__...
      |    |       |     
      |    |       |__val
      |    |            |__annotations
      |    |            |     ...
      |    |            |__binary_annotations
      |    |            |     ...
      |    |            |__class_embeddings_h
      |    |            |     ...
      |    |            |__sam_features_h
      |    |                  ...
      |                   
      |__ckp
      |    |__sam
      |    |   |__sam_vit_h_4b8939.pth
      |    |
      |    |__surgical_sam
      |            |__endovis_2018
      |            |     ...
      |            |__endovis_2017
      |                    |__fold0
      |                    |     ...
      |                    |__fold1
      |                    |     ...
      |                    |__fold2
      |                    |     ...
      |                    |__fold3
      |                          ...
      |   
      |__segment_anything
      |    ...
      |__surgicalSAM
           ...
  ```

##  Train
To train the model:
```
cd surgicalSAM/
python train.py  --dataset endovis_2018
```

##  Inference
To run inference on our provided SurgicalSAM checkpoints and obtain evaluation results:
```
cd surgicalSAM/
python inference.py  --dataset endovis_2018
```

##  Citing SurgicalSAM

```
@article{yue_surgicalsam,
  title={SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation},
  author={Yue, Wenxi and Zhang, Jing and Hu, Kun and Xia, Yong and Luo, Jiebo and Wang, Zhiyong},
  booktitle={AAAI},
  year={2024}
}
```

##  Citing SurgicalSAM

```
@inproceedings{zhao2024sfc,
  title={Sfc: Shared feature calibration in weakly supervised semantic segmentation},
  author={Zhao, Xinqiao and Tang, Feilong and Wang, Xiaoyang and Xiao, Jimin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7525--7533},
  year={2024}
}
```


##  References
[1]Wenxi Yue, Jing Zhang, Kun Hu, Yong Xia, Jiebo Luo, and Zhiyong Wang. Surgicalsam:

Efficient class promptable surgical instrument segmentation. In *Proceedings of the AAAI*

*Conference on Artificial Intelligence*, volume 38, pages 6890–6898, 2024.

[2] Xinqiao Zhao, Feilong Tang, Xiaoyang Wang, and Jimin Xiao. Sfc: Shared feature calibra

tion in weakly supervised semantic segmentation. In *Proceedings of the AAAI Conference*

*on Artificial Intelligence*, volume 38, pages 7525–7533, 2024.
