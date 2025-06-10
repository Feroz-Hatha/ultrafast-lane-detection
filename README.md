# Ultra-Fast Lane Detection Using Structure Aware Deep Learning

This repository contains a complete implementation of the **Ultra Fast Structure-aware Deep Lane Detection** model, as described in the [ECCV 2020 paper](https://arxiv.org/abs/2004.11757), using the **TuSimple** lane detection dataset.

![Sample Prediction](outputs/pred_23.png)

---

## Project Highlights

* Full pipeline built from scratch in PyTorch (data loading → training → evaluation)
* Custom loss function combining classification + structural constraints (L\_sim + L\_shp)
* Achieves **77.42% accuracy** on the TuSimple validation set
* Ground truth and predictions can be visualized using lane anchor mapping

---

## 📁 Directory Structure

```
ultrafast-lane-detection/
├── datasets/
│   └── tusimple.py                  # TuSimple dataset class
├── models/
│   └── ultrafast_lane_net.py       # Model architecture (ResNet backbone)
├── utils/
│   ├── dataloader_utils.py         # Custom collate function for dataloader
│   ├── visualize_labels.py         # Lane labels visualization
│   ├── losses.py                   # Structural loss components
│   ├── split_test_labels.py        # Genrate splits for validation and test
│   └── target_generator.py         # Prediction visualization
├── train.py                        # Training loop with structured loss
├── evaluate.py                     # Evaluation script (outputs accuracy and visualizations)
├── visualize_predictions.ipynb     # Visualize predicted lanes
├── config.yaml                     # Config file to assign various parameters
├── checkpoints/                    # Trained models saved by epoch or best
├── outputs/                        # Sample prediction visualizations
└── README.md
```

---

## 📦 Requirements

* Python 3.8+
* PyTorch 1.12+
* torchvision
* numpy
* matplotlib
* OpenCV
* tqdm
* PIL

---

## TuSimple Dataset Setup

1. Download the TuSimple dataset from [Kaggle](https://www.kaggle.com/datasets/manideep1108/tusimple).
2. Extract the contents so that your folder structure looks like:

```
TUSimple/
├── train_set/
│   ├── clips/
│   ├── label_data_*.json
├── test_set/
│   ├── clips/
│   ├── test_label.json
```

3. Run the command `cat label_data_0313.json label_data_0531.json label_data_0601.json > train_label.json` from the `train_set` folder to combine the training labels.

4. Run the script `utils/split_test_labels.py` to create a validation split:

```bash
python utils/split_test_labels.py
```

---

## Training

Train the model using:

```bash
python train.py
```

The script will save the best model checkpoint to `checkpoints/best_model.pth`.

---

## Evaluation

Evaluate on the validation split:

```bash
python evaluate.py
```

Outputs:

* TuSimple accuracy score
* Lane prediction visualizations under `outputs/`

---

## 🛠️ Ongoing / Planned Enhancements

* [ ] Add segmentation branch for auxiliary supervision
* [ ] Incorporate ONNX export or TorchScript tracing
* [ ] Track experiments using Weights & Biases (W\&B)
* [ ] Optimize inference pipeline with post-processing

---

## 📊 Results (so far)

| Metric       | Value     |
| ------------ | --------- |
| Accuracy     | 77.42%    |
| Model        | ResNet-18 |
| Input Size   | 360×640   |
| Gridding Num | 100       |
| Row Anchors  | 56        |

---

## 📖 Reference

> **Ultra Fast Structure-aware Deep Lane Detection**
> Zequn Qin, Huanyu Wang, Xi Li
> *ECCV 2020*
> [arXiv:2004.11757](https://arxiv.org/abs/2004.11757)