# TrackNetV4

<p align="center">
  <img src="assets/pipeline.jpg" width="900" alt="TrackNetV4 Pipeline"/>
  <br>
  &nbsp;&nbsp; 🌐 <a href="https://TrackNetV4.github.io/">Project Website</a> ｜ 📑 <a href="https://ieeexplore.ieee.org/document/10889364">Paper</a>
</p>



This repository contains the official TensorFlow implementation of our ICASSP 2025 paper: **TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps**.

We provide TensorFlow code demonstrating our Motion-Aware Fusion framework, integrated with TrackNetV2, for training, testing, and visual prediction of sports object trajectories. Our implementation supports three datasets: tennis, badminton, and newly introduced badminton dataset. Additionally, we provide model code for integrating the framework with TrackNetV3.

Additionally, feel free to explore the project website for more visualizations, dataset details, and experimental results.


## 🚀 Getting Started

### 📥 1. Clone the Repository

```bash
git clone https://github.com/AR4152/TrackNetV4.git
cd TrackNetV4
```

### 🧪 2. Set Up the Environment

We recommend using [**conda**](https://anaconda.org/anaconda/conda) for environment management.

```bash
conda env create -f environment.yml
conda activate TrackNetV4
```

💡 *Tip: Rename the environment if you prefer: `-n your_env_name` in the command above.*

### 🗂️ 3. Prepare the Dataset

Refer to the dataset preparation instructions inside [`docs/DATASET.md`](docs/DATASET.md).

### ⬇️ 4. Download Pretrained Models (Optional)

We provide several pretrained model checkpoints for evaluation and fine-tuning. For download links and performance metrics, see [`docs/RESULT.md`](docs/RESULT.md).

## Usage

### 🔧 Training

The `train.py` script is used to train a TrackNet model (e.g., Baseline_TrackNetV2, TrackNetV4 variants) on a selected dataset. It supports configurable training parameters, optional pretrained weight loading, checkpoint saving, and intermediate evaluation.

**Command**

```bash
python src/train.py --model_name <MODEL> --dataset <DATASET> [options]
```

**🔁 Example**

```bash
python src/train.py --model_name Baseline_TrackNetV2 \
    --dataset tennis_game_level_split \
    --batch_size 2 \
    --learning_rate 1.0 \
    --height 288 \
    --width 512 \
    --epochs 30 \
    --tol 4 \
    --work_dir ./models \
    --save_freq 1
```

<details>
  <summary>📝 Arguments</summary>


| Argument        | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `--model_name`  | Name of the model to train. Choices: `Baseline_TrackNetV2`, `TrackNetV4_TypeA`, `TrackNetV4_TypeB`. **(Required)** |
| `--dataset`     | Dataset to use. Choices: `tennis_game_level_split`, `tennis_clip_level_split`, `badminton`, `new_tennis`. **(Required)** |
| `--batch_size`  | Batch size for training. *(Default: 2)*                                     |
| `--learning_rate` | Learning rate for Adadelta optimizer. *(Default: 1.0)*                   |
| `--height`      | Image height. *(Default: 288)*                                              |
| `--width`       | Image width. *(Default: 512)*                                               |
| `--epochs`      | Number of training epochs. *(Default: 30)*                                  |
| `--tol`         | Tolerance threshold for evaluation. *(Default: 4)*                          |
| `--model_path`   | Path to a `.keras` pretrained model to fine-tune. *(Optional)*              |
| `--work_dir`    | Directory to save models. *(Default: `./models`)* If default, adds a timestamp. |
| `--save_freq`   | Save model checkpoint every N epochs. *(Default: 1)*                        |


</details>



**🧠 Notes**

- Saved model checkpoints are stored in the specified `--work_dir`.
- A final model will be saved as `model_final.keras` after the last epoch.
- If you're training from a pretrained model (`--model_path`), weights will be loaded and adapted to the new architecture using `skip_mismatch=True`.


### 📊 Evaluation

The `eval.py` script is used to evaluate a trained TrackNet model on a selected test dataset. It calculates key metrics such as accuracy, precision, recall, F1 score, and inference speed. Results are saved in a structured JSON file for later analysis.

**Command**

```bash
python src/eval.py --model_path <MODEL_PATH> --dataset <DATASET> [options]
```

**🔁 Example**

```bash
python src/eval.py --model_path ./models/model_final.keras \
    --dataset tennis_game_level_split \
    --batch_size 2 \
    --tol 4 \
    --result_dir ./results
```

<details>
  <summary>📝 Arguments</summary>

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--model_path`   | Path to the trained `.keras` model to evaluate. **(Required)**              |
| `--dataset`      | Dataset to evaluate on. Choices: `tennis_game_level_split`, `tennis_clip_level_split`, `badminton`, `new_tennis`. **(Required)** |
| `--batch_size`   | Batch size for evaluation. *(Default: 2)*                                   |
| `--tol`          | Tolerance threshold for evaluation metrics. *(Default: 4)*                  |
| `--result_dir`   | Directory to save the JSON result file. *(Default: `./`)*                   |

</details>

**🧠 Notes**

- Evaluation metrics include: True Positives, True Negatives, False Positives (FP1, FP2), False Negatives, Accuracy, Precision, Recall, F1 Score, and Inference Speed (FPS).
- Results are saved as a `.json` file named after the evaluated model (e.g., `model_final.json`).
- The evaluation includes batch-wise logging of prediction outcomes.
- The script supports all model variants that rely on custom layers such as `MotionPromptLayer` and `FusionLayerTypeA/B`.


### 🎯 Prediction

The `predict.py` script performs trajectory prediction on a video using a trained TrackNet model. It processes frames, runs inference, and outputs both an annotated video and a CSV file containing predicted coordinates.

**Command**

```bash
python src/predict.py --video_path <VIDEO> --model_weights <MODEL_PATH> --output_dir <OUTPUT_DIR> [options]
```

**🔁 Example**

```bash
python src/predict.py --video_path ./videos/sample_match.mp4 \
    --model_weights ./models/model_final.keras \
    --output_dir ./predictions \
    --queue_length 10
```

<details>
  <summary>📝 Arguments</summary>

| Argument           | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `--video_path`     | Path to the input video file. Supported formats: `.avi`, `.mp4`. **(Required)** |
| `--model_weights`  | Path to the `.keras` model file with trained weights. **(Required)**         |
| `--output_dir`     | Directory to save the output video and CSV file. *(Default: current dir)*   |
| `--queue_length`   | Number of past trajectory points to display. *(Default: 5)*                 |

</details>

**📦 Output**

- A **CSV file** containing predicted (X, Y) positions per frame.
- An **annotated video** showing the trajectory with past predictions (green) and current (red).

**🧠 Notes**

- The model uses a sliding window of 3 consecutive frames for each prediction.
- Predicted points are visualized using bounding boxes and stored using the format:
  ```
  Frame, Visibility, X, Y
  ```
- `queue_length` controls how many historical predictions are shown in the trajectory.
- The model supports advanced TrackNetV4 variants using custom layers like `MotionPromptLayer`, `FusionLayerTypeA/B`, etc.


## Citation

If you find TrackNetV4 useful in your research, please consider 📝 citing our paper and ⭐️ star our repo:

```BibTeX
@INPROCEEDINGS{10889364,
  author={Raj, Arjun and Wang, Lei and Gedeon, Tom},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Visualization;Tracking;Sports equipment;Signal processing;Predictive models;Trajectory;Object tracking;Speech processing;Sports;Videos;tracking;motion attention;fusion},
  doi={10.1109/ICASSP49660.2025.10889364}}
```

## Acknowledgment

This codebase is built on top of [VMPs](https://github.com/q1xiangchen/VMPs/), and [TrackNetV2](https://gitlab.nol.cs.nycu.edu.tw/open-source/TrackNetv2/) and we thank the authors for their work.
