# Multitask Learning Strengthens Adversarial Robustness

```
@inproceedings{mao2020multitask,
  author    = {Chengzhi Mao and
               Amogh Gupta and
               Vikram Nitin and
               Baishakhi Ray and
               Shuran Song and
               Junfeng Yang and
               Carl Vondrick},
  title     = {Multitask Learning Strengthens Adversarial Robustness},
  booktitle = {Computer Vision - {ECCV} 2020 - 16th European Conference, Glasgow,
               UK, August 23-28, 2020, Proceedings, Part {II}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12347},
  pages     = {158--174},
  publisher = {Springer},
  year      = {2020},
  url       = {https://doi.org/10.1007/978-3-030-58536-5\_10},
  doi       = {10.1007/978-3-030-58536-5\_10},
}
```

# Demo for Robustness under multitask attack

Download Cityscapes dataset from [Cityscapes](https://www.cityscapes-dataset.com/downloads/). 

Download pretrained DRN-22 model from [DRN model zoo](https://drive.google.com/drive/folders/0B_4LoEXGO1TwcmhzLXpWUVFEMXM).

Modify the path to data and model in `demo_mtlrobust.py`.

Run demo to see the trend that model overall robustness is increased when the output dimension increased.

To see the gradient norm measurement of robustness, set `get_grad=True`,

To see the actually robust accuracy for model, set `test_acc_output_dim=False`

`python demo_mtlrobust.py`

which explains why segmentation is inherently robust.




# CityScape

## Data preprocessing

Run `python data_resize_cityscape.py` to resize to smaller images.

## Train Robust model against single task attack
1. Set up the path to data in `config/drn_d_22_cityscape_config.json`

2. Run `cityscape_example.sh` to train a main task with auxiliary task for robustness.

# Taskonomy

## Data Preprocessing

You can use our preprocessed data from [preprocessed data](https://cv.cs.columbia.edu/mcz/taskonomy_small.zip)

Or do from scratch

1. Download data from [official raw data](https://github.com/alexsax/taskonomy-sample-model-1).

1. Run `python data_resize_taskonomy.py` to resize to smaller images.

2. Rename `segment_semantic` to `segmentsemantic`.



## Train Robust model against single task attack
1. Set up the path to data in `config/resnet18_taskonomy_config.json`

2. Run `taskonomy_example.sh` to train a main task with auxiliary task for robustness. For different task, we have different
different setup, refer to our paper and supplementary for details.


## Model evaluation

We offer our pretrained models to download here: Cityscapes [segmentation](https://cv.cs.columbia.edu/mcz/city_seg_mtl.zip)
 [depth](https://cv.cs.columbia.edu/mcz/city_depth_mtl.zip)
and Taskonomy [taskonomy segmentation demo](https://cv.cs.columbia.edu/mcz/taskonomy_seg_demo.zip)


After setting up the path to your downloaded models in `test_cityscapes_seg.py` and `test_taskonomy_seg.py`,

Run `python test_cityscapes_seg.py` and `python test_taskonomy_seg.py` for evaluating the robustness of multitask models under
single task attacks.

Pretrained models for other tasks for Taskonomy can be downloaded [here, comming soon](comming soon)

# Acknowledgement

Our code refer the code at:
https://github.com/fyu/drn/blob/master/drn.py
Taskonomy
https://github.com/tstandley/taskgrouping, 

We thank the authors for open sourcing their code.
