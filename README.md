# SegPVSG: Panoptic Video Scene Graph Generation via Temporal Focusing and Generative Augmentation (ICML 2026)

## Authors

**Yikai Li**<sup>* 1</sup>**, Quhui Ke**<sup>* 1</sup>**, Jinglin Liang**<sup>1</sup>**, Zhiyuan Zhang**<sup>1</sup>**, Zhidi Lin**<sup>2</sup>**, Shuangping Huang**<sup>† 1 3</sup>

<sup>1</sup> South China University of Technology, Guangzhou, China  
<sup>2</sup> The University of Hong Kong, Hong Kong SAR, China  
<sup>3</sup> Pazhou Laboratory, Guangzhou, China  

<sup>*</sup> Equal contribution  <sup>†</sup> Corresponding author

## Paper and Links

- Paper (ICML 2026): [SegPVSG](https://openreview.net/forum?id=nljvkUyZRy)

## Updates

- [08/2026] Initial public code release for TFN.

---

## The PVSG Dataset

Please refer to [OpenPVSG](https://github.com/LilyDaytoy/OpenPVSG) for data preparation. The `data` directory should be placed under the project root:

```text
data
├── ego4d
│   ├── frames
│   ├── masks
│   └── videos
├── epic_kitchen
│   ├── frames
│   ├── masks
│   └── videos
├── vidor
│   ├── frames
│   ├── masks
│   └── videos
└── pvsg.json
```

## Environment

### System Environment

The code has been tested in the following system environment:

- OS: Ubuntu 18.04.6
- glibc: 2.27
- CUDA: 12.0
- GPU: NVIDIA GeForce RTX 3090
- NVIDIA Driver: 525.105.17

### Python Dependencies

```text
python 3.9
torch 1.10.2
torchvision 0.11.3
mmcv-full 1.6.0
mmdet 2.25.0
numpy 1.23.5
opencv-python 4.11.0.86
setuptools 68.2.2
imageio 2.35.1
lap 0.5.12 
cython_bbox 0.1.5 
scipy 1.10.1
einops 0.8.2
tqdm
# Required only when logging.use_swanlab is true.
swanlab
```

## Feature Preparation

Please refer to [OpenPVSG](https://github.com/LilyDaytoy/OpenPVSG) for the Mask2Former and UniTrack checkpoints, and place them in the `checkpoints` directory.


### IPS + Tracking

```bash
# Extract visual features for train set
bash scripts/utils/prepare_feats_ips.sh
# Prepare relation set for train set
bash scripts/utils/prepare_rel_set.sh --gt-time
# Extract visual features for val set
bash scripts/utils/prepare_feats_ips.sh --split val --work-dir outputs/features/ips_val_vf
# Prepare relation set for val set
bash scripts/utils/prepare_rel_set.sh --split val --work_dir outputs/features/ips_val_vf
```

### VPS

```bash
# Extract visual features for train set
bash scripts/utils/prepare_feats_vps.sh
# Prepare relation set for train set
bash scripts/utils/prepare_rel_set.sh --work_dir outputs/features/vps_train_vf --gt-time
# Extract visual features for val set
bash scripts/utils/prepare_feats_vps.sh --split val --work-dir outputs/features/vps_val_vf
# Prepare relation set for val set
bash scripts/utils/prepare_rel_set.sh --split val --work_dir outputs/features/vps_val_vf
```

To speed up feature extraction, you can partition the videos by index, for example:

```bash
bash scripts/utils/prepare_feats_ips.sh --gpu 0 --video-start 0 --video-step 4
bash scripts/utils/prepare_feats_ips.sh --gpu 1 --video-start 1 --video-step 4
bash scripts/utils/prepare_feats_ips.sh --gpu 2 --video-start 2 --video-step 4
bash scripts/utils/prepare_feats_ips.sh --gpu 3 --video-start 3 --video-step 4
```

## Evaluation

> Evaluation metrics may differ from those reported due to code and environment differences.  
> Download the pretrained models from [SegPVSG Demo Checkpoints](https://github.com/ticatt/SegPVSG/releases/download/v1.0.0/segpvsg_demo_checkpoints.zip).

```bash
bash scripts/eval/eval_tfn.sh --exp-name demo_segpvsg_ips --epoch epoch_best
bash scripts/eval/eval_tfn.sh --exp-name demo_segpvsg_vps --epoch epoch_best
bash scripts/eval/eval_tfn.sh --exp-name demo_tfn_ips --epoch epoch_best
bash scripts/eval/eval_tfn.sh --exp-name demo_tfn_vps --epoch epoch_best
```

## Training

> Performance is sensitive to the random seed and training epoch.

```bash
# Train TFN in IPS setting
bash scripts/train/train_tfn.sh --config configs/tfn/train_ips.yaml --exp-name tfn_ips

# Train TFN in VPS setting
bash scripts/train/train_tfn.sh --config configs/tfn/train_vps.yaml --exp-name tfn_vps
```

## Citation

```bibtex
@inproceedings{li2026segpvsg,
  title     = {{SegPVSG}: Panoptic Video Scene Graph Generation via Temporal Focusing and Generative Augmentation},
  author    = {Li, Yikai and Ke, Quhui and Liang, Jinglin and Zhang, Zhiyuan and Lin, Zhidi and Huang, Shuangping},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
