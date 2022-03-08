# Out-of-distribution Generalization Investigation on Vision Transformers
This repository contains PyTorch evaluation code for _CVPR 2022_ accepted paper [Delving Deep into the Generalization of Vision Transformers under Distribution Shifts](https://arxiv.org/abs/2106.07617).

## Taxonomy of Distribution Shifts
<p align="middle">
<img src="https://github.com/Phoenix1153/ViT_OOD_generalization/raw/main/img/taxonomy.png" width="80%">
<p>

**Illustration of our taxonomy of distribution shifts.** We build the taxonomy upon what kinds of semantic concepts are modified from the original image and divide the distribution shifts into four cases: background shifts, corruption shifts, texture shifts, and style shifts. <img src="http://latex.codecogs.com/gif.latex?{\color{Red} \checkmark}" /> denotes the unmodified vision cues under certain type of distribution shifts. Please refer to the literature for details.

### Datasets Used for Investigation
- **Background Shifts.** [ImageNet-9](https://github.com/MadryLab/backgrounds_challenge/releases) is adopted for background shifts. ImageNet-9 is a variety of 9-class datasets with different foreground-background recombination plans, which helps disentangle the impacts of foreground and background signals on classification. In our case, we use the four varieties of generated background with foreground unchanged, including 'Only-FG', 'Mixed-Same', 'Mixed-Rand' and 'Mixed-Next'. The 'Original' data set is used to represent in-distribution data.
- **Corruption Shifts.** [ImageNet-C](https://zenodo.org/record/2235448#.YMwT5JMzalY) is used to examine generalization ability under corruption shifts. ImageNet-C includes 15 types of algorithmically generated corruptions, grouped into 4 categories: ‘noise’, ‘blur’, ‘weather’, and ‘digital’. Each corruption type has five levels of severity, resulting in 75 distinct corruptions.
- **Texture Shifts.** [Cue Conflict Stimuli](https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/style-transfer-preprocessed-512) and [Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet) are used to investigate generalization under texture shifts. Utilizing style transfer, Geirhos et al. generated Cue Conflict Stimuli benchmark with conflicting shape and texture information, that is, the image texture is replaced by another class with other object semantics preserved. In this case, we respectively report the shape and texture accuracy of classifiers for analysis. Meanwhile, Stylized-ImageNet is also produced in Geirhos et al. by replacing textures with the style of randomly selected paintings through AdaIN style transfer.
- **Style Shifts.** [ImageNet-R](https://github.com/hendrycks/imagenet-r) and [DomainNet](http://ai.bu.edu/M3SDA/) are used for the case of style shifts. ImageNet-R contains 30000 images with various artistic renditions of 200 classes of the original ImageNet validation data set. The renditions in ImageNet-R are real-world, naturally occurring variations, such as paintings or embroidery, with textures and local image statistics which differ from those of ImageNet images. DomainNet is a recent benchmark dataset for large-scale domain adaptation that consists of 345 classes and 6 domains. As labels of some domains are very noisy, we follow the 7 distribution shift scenarios in Saito et al. with 4 domains (Real, Clipart, Painting, Sketch) picked.

## Generalization-Enhanced Vision Transformers
<p align="middle">
<img src="https://github.com/Phoenix1153/ViT_OOD_generalization/raw/main/img/new_DANN-1.png" width="40%">
<img src="https://github.com/Phoenix1153/ViT_OOD_generalization/raw/main/img/new_MME-1.png" width="45%">
<p>
<p align="middle">
<img src="https://github.com/Phoenix1153/ViT_OOD_generalization/raw/main/img/new_SSL-1.png" width="90%">
<p>

**A framework overview of the three designed generalization-enhanced ViTs.** All networks use a Vision Transformer <img src="http://latex.codecogs.com/gif.latex?F" /> as feature encoder and a label prediction head <img src="http://latex.codecogs.com/gif.latex?C" /> . Under this setting, the inputs to the models have labeled source examples and unlabeled target examples. **top left:** **T-ADV** promotes the network to learn domain-invariant representations by introducing a domain classifier <img src="http://latex.codecogs.com/gif.latex?D" /> for domain adversarial training. **top right:** **T-MME** leverage the minimax process on the conditional entropy of target data to reduce the distribution gap while learning discriminative features for the task. The network uses a cosine similarity-based classifier architecture <img src="http://latex.codecogs.com/gif.latex?C" />  to produce class prototypes. **bottom:** **T-SSL** is an end-to-end prototype-based self-supervised learning framework. The architecture uses two memory banks <img src="http://latex.codecogs.com/gif.latex?V^s" /> and <img src="http://latex.codecogs.com/gif.latex?V^t" /> to calculate cluster centroids. A cosine classifier <img src="http://latex.codecogs.com/gif.latex?C" />  is used for classification in this framework.


## Run Our Code
## Environment Installation
>     conda create -n vit python=3.6
>     conda activate vit
>     conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch

## Before Running

>     conda activate vit
>     PYTHONPATH=$PYTHONPATH:.

## Evaluation

>     CUDA_VISIBLE_DEVICES=0 python main.py \
>     --model deit_small_b16_384 \
>     --num-classes 345 \
>     --checkpoint data/checkpoints/deit_small_b16_384_baseline_real.pth.tar \
>     --meta-file data/metas/DomainNet/sketch_test.jsonl \
>     --root-dir data/images/DomainNet/sketch/test

## Experimental Results

### DomainNet

#### DeiT\_small\_b16\_384

confusion matrix for the baseline model

|          | clipart | painting | real  | sketch |
| -------- | ------- | -------- | ----- | ------ |
| clipart  | 80.25   | 33.75    | 55.26 | 43.43  |
| painting | 36.89   | 75.32    | 52.08 | 31.14  |
| real     | 50.59   | 45.81    | 84.78 | 39.31  |
| sketch   | 52.16   | 35.27    | 48.19 | 71.92  |

Above used models could be found [here](https://drive.google.com/drive/folders/1XUAIQ-TNQaG7FuvqG-Qzptn2puyGFFoQ?usp=sharing).


### Remarks

- These results may slightly differ from those in our paper due to differences of the environments.

- We will continuously update this repo.

## Citation
If you find these investigations useful in your research, please consider citing:
```
@misc{zhang2021delving,  
      title={Delving Deep into the Generalization of Vision Transformers under Distribution Shifts}, 
      author={Chongzhi Zhang and Mingyuan Zhang and Shanghang Zhang and Daisheng Jin and Qiang Zhou and Zhongang Cai and Haiyu Zhao and Shuai Yi and Xianglong Liu and Ziwei Liu},  
      year={2021},  
      eprint={2106.07617},  
      archivePrefix={arXiv},  
      primaryClass={cs.CV}  
}
```

