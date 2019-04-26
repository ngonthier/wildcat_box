# wildcat.pytorch
PyTorch implementation of "WILDCAT: Weakly Supervised Learning of Deep ConvNets for Image Classification, Pointwise Localization and Segmentation", CVPR 2017 (http://webia.lip6.fr/~durandt/pdfs/2017_CVPR/Durand_WILDCAT_CVPR_2017.pdf) reprise par Nicolas pour produire des boites et fonctionner sur IconArt_v1

### Requirements
Please, install the following packages
- numpy
- torch
- torchnet
- torchvision
- tqdm

Possibilité d'utiliser pour installer trochnet : pip install git+https://github.com/pytorch/tnt.git@master

### Options

- `k`: number of regions for the spatial pooling. If `k` is larger than 1, `k` is the number of regions, otherwise `k` is the proportion of selected regions. `k=0.2` means that 20% of the regions are used.  
- `maps`: number of maps for each class
- `alpha`: weight for minimum regions
- `lr`: learning rate
- `lrp`: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is `lr * lrp`
- `batch-size`: number of images per batch
- `image-size`: size of the image
- `epochs`: number of training epochs

### Demo VOC 2007
```sh
python3 -m wildcat.demo_voc2007 ../data/voc --image-size 448 --batch-size 16 --lrp 0.1 --lr 0.01 --epochs 20 --k 0.2 --maps 8 --alpha 0.7
```

### Demo MIT67
```sh
python3 -m wildcat.demo_mit67 ../data/mit67 --image-size 448 --batch-size 16 --lrp 0.1 --lr 0.001 --epochs 20 --k 0.4 --maps 8
```

### Demo IconArt_v1
For training :
```sh
python3 -m wildcat.demo_IconArt_v1 ../data/voc --image-size 600 --batch-size 8 --lrp 0.1 --lr 0.01 --epochs 20 --k 0.2 --maps 4 --alpha 0.7
python3 -m wildcat.demo_IconArt_v1 ../data/voc --image-size 600 --batch-size 8 --lrp 0.1 --lr 0.01 --epochs 20 --k 25 --maps 8 --alpha 0.7
```

For testing and plot images boxes
```sh
python3 -m wildcat.demo_IconArt_v1 ../data/voc --image-size 600 --batch-size 8 --lrp 0.1 --lr 0.01 --epochs 20 --k 0.2 --maps 4 --alpha 0.7 --test --plot
```

## Citing this repository

If you find this code useful in your research, please consider citing us:

```
@inproceedings{Durand_WILDCAT_CVPR_2017,
author = {Durand, Thibaut and Mordan, Taylor and Thome, Nicolas and Cord, Matthieu},
title = {{WILDCAT: Weakly Supervised Learning of Deep ConvNets for Image Classification, Pointwise Localization and Segmentation}},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2017}
}
```

## Licence

MIT License
