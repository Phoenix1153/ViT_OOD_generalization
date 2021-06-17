# Out-of-distribution Generalization Investigation on Vision Transformers
This repository contains PyTorch evaluation code for [Delving Deep into the Generalization of Vision Transformers under Distribution Shifts](https://arxiv.org/abs/2106.07617).

## Quick Glance of Our Works
<p align="middle">
<img src="https://github.com/Phoenix1153/ViT_OOD_generalization/raw/main/img/overall-1.png" width="48%">
<img src="https://github.com/Phoenix1153/ViT_OOD_generalization/raw/main/img/DA-1.png" width="48%">
<p>
<p align="middle">"(a)                     (b)"</p>

**A quick glance of our investigation observations.** (a) Investigation of IID/OOD Generalization Gap implies that ViTs generalize better than CNNs under most types of distribution shifts. (b) Combined with generalization-enhancing methods, we achieve significant performance boosts on the OOD data by 4\% compared with vanilla ViTs, and consistently outperform the corresponding CNN models. The enhanced ViTs also have smaller IID/OOD Generalization Gap than the ehhanced BiT models.

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

