
# 📄 Results & Experimental Details

This document outlines the results, model weights, and experimental setup used in our study.

## ✅ Results Overview

### 🎾 Tennis Dataset  

**Game Level Protocol**

| Model                          | Total Test Images | TP    | TN   | FP1 | FP2 | FN  | Acc. | Prec. | Rec.  | F1    | Speed    | Model Weights Link |
|-------------------------------|-------------------|-------|------|-----|-----|-----|------|--------|--------|--------|----------|---------------------|
| TrackNetV2                    | 17193             | 15863 | 396  | 142 | 17  | 775 | 94.6 | 99.0   | 95.7   | 97.3   | 156.9 FPS| [Download](#)       |
| TrackNetV2 (+Motion)          | 17193             | 15973 | 389  | 167 | 24  | 640 | 95.2 | 98.8   | 96.1   | 97.5   | 155.7 FPS| [Download](#)       |

**Clip Level Protocol**

| Model                          | Total Test Images | TP    | TN   | FP1 | FP2 | FN  | Acc. | Prec. | Rec.  | F1    | Speed    | Model Weights Link |
|-------------------------------|-------------------|-------|------|-----|-----|-----|------|--------|--------|--------|----------|---------------------|
| TrackNetV2                    | 17760             | 16195 | 393  | 163 | 25  | 993 | 93.4 | 98.9   | 94.2   | 96.4   | 160.9 FPS| [Download](#)       |
| TrackNetV2 (+Motion)          | 17760             | 16243 | 399  | 199 | 19  | 778 | 94.4 | 98.7   | 95.5   | 97.0   | 158.6 FPS| [Download](#)       |


### 🏸 Badminton Dataset  

| Model                           | Total Test Images | TP    | TN    | FP1  | FP2  | FN   | Acc. | Prec. | Rec.  | F1    | Speed     | Model Weights Link |
|--------------------------------|-------------------|-------|-------|------|------|------|------|--------|--------|--------|-----------|---------------------|
| YOLOv7                         | 13064             | 9447  | 1514  | 751  | 218  | 1134 | 72.3 | 78.5   | 60.0   | 68.0   | 2.9 FPS   | [Download](#)       |
| TrackNetV2 (3 in 1 out)        | 39192             | 29129 | 4264  | 468  | 358  | 4973 | 85.2 | 92.2   | 85.4   | 88.6   | 31.8 FPS  | [Download](#)       |
| TrackNetV2 (3 in 3 out)        | 37794             | 26324 | 6013  | 438  | 393  | 4526 | 85.6 | 92.0   | 85.3   | 88.5   | 33.9 FPS  | [Download](#)       |
| TrackNetV2                     | 37794             | 26529 | 5993  | 523  | 511  | 4731 | 84.6 | 90.8   | 84.8   | 87.7   | 162.1 FPS | [Download](#)       |
| TrackNetV2 (+Motion)           | 37794             | 26878 | 5834  | 565  | 672  | 4845 | 86.6 | 90.7   | 85.1   | 87.8   | 161.1 FPS | [Download](#)       |
| TrackNetV3 (3 in 3 out)        | 10836             | 8869  | 1378  | 55   | 66   | 468  | 94.6 | 99.2   | 95.0   | 96.8   | 15.0 FPS  | [Download](#)       |
| TrackNetV3 (+Motion)           | 10836             | 8919  | 1387  | 55   | 16   | 459  | 95.1 | 99.2   | 95.1   | 97.1   | **15.0\*** FPS | [Download](#)       |


## 🧪 Experimental Setup

All experiments were conducted under consistent conditions to ensure fair and reliable comparisons across all models and datasets.


## 📐 Input Configuration

- **Input image resolution**: `512 × 288`  
- **Input frame sequence**: `3` consecutive frames  
- **Output prediction**: `3` heatmaps (corresponding to each input frame)

## 📊 Baseline Models

Baseline models were implemented using **TrackNetV2** and **TrackNetV3** (tracking module only):

- **Learning rate**: `1.0`  
- **Training duration**: `30 epochs`  
- **Model selection criteria**: Best-performing checkpoint on validation data

## 🔁 TrackNetV4 Fine-Tuning

We enhanced the baselines with our **motion-aware fusion framework**, applied to **TrackNetV4**:

- **Initialization**: Pretrained weights from TrackNetV2/TrackNetV3  
- **Learning rates explored**: `1e-3`, `1e-4`, `1e-5`  
- **Training duration**: `30 epochs`  
- **Model selection criteria**: Best-performing checkpoint on validation data

