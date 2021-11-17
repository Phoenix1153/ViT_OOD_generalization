# Out-of-distribution Generalization Investigation on Vision Transformers

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


