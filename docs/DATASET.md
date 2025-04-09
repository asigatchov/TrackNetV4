# 📂 Dataset Setup

This repository supports three datasets out of the box and also allows integration with custom datasets via a provided base dataset class.

<p align="center">
  <a href="#-tennis-dataset">🎾 Tennis Dataset</a> |
  <a href="#-badminton-dataset">🏸 Badminton Dataset</a> |
  <a href="#-new-tennis-dataset">🚀🎾 New Tennis Dataset</a> |
  <a href="%EF%B8%8F-customising-dataset">⚙️ Customising Dataset</a>
</p>

## 🎾 Tennis Dataset

- Introduced in [TrackNet](https://arxiv.org/abs/1907.03698) paper.

### 📊 Evaluation Protocols

The original dataset lacks a standardized training and test split. To ensure consistency in evaluation, we define two custom protocols while maintaining an approximate 70/30 frame split:

1. **Game-level Split**  
    Entire games are assigned to either the training or test set.  
    - **Training games**: `game2`, `game3`, `game5`, `game6`, `game7`, `game8`, `game10`  
    - **Test games**: `game1`, `game4`, `game9`  
    This setup results in approximately **70.81%** of the total frames used for training and **29.19%** for testing.
    - 📄 CSV: [`data/splits/tennis_game_level_split.csv`](/data/splits/tennis_game_level_split.csv)

2. **Clip-level Split**  
    The dataset is originally structured into multiple clips per game. For this protocol:  
    - Clips are split based on cumulative frame count to maintain the **70/30** frame ratio.  
    - Each clip is wholly assigned to either the training or test set, ensuring no overlap.
    - 📄 CSV: [`data/splits/tennis_clip_level_split.csv`](/data/splits/tennis_clip_level_split.csv)

### 🔧 Setup Instructions

1. Download the dataset using the Google Drive link provided in this [repository](https://github.com/yastrebksv/TrackNet).
2. Unzip the `Dataset.zip` file.
3. Move the extracted `Dataset` folder into the `data/tennis/` directory as shown below:
    <details>
      <summary>Expected Directory Structure (Before Preprocessing)</summary>
    
      <pre>
    repo-root/
    └── data/
        └── tennis/
            └── Dataset/
                ├── game1/
                │   ├── Clip1/
                │   │   ├── 0000.jpg
                │   │   ├── 0001.jpg
                │   │   ├── 0002.jpg
                │   │   ├── 0003.jpg
                │   │   ├── 0004.jpg
                │   │   ├── 0005.jpg
                │   │   └── ...
                │   ├── Clip2/
                │   ├── Clip3/
                │   ├── ...
                │   └── Clip10/
                ├── game2/
                ├── game3/
                ├── ...
                └── game10/
      </pre>
    </details>

4. Run the commands below to preprocess the data:
    - For Game-level Split:
    
      ```bash
      python3 scripts/preprocess.py --dataset tennis_game_level_split
      ```
    
    - For Clip-level Split:
    
      ```bash
      python3 scripts/preprocess.py --dataset tennis_clip_level_split
      ```

    <details>
      <summary>Expected Directory Structure (After Preprocessing)</summary>
    
      <pre>
    repo-root/
    └── data/
        └── tennis/
            └── Dataset/
                ├── game1/
                │   ├── Clip1/
                │   │   ├── 0000.jpg
                │   │   ├── 0001.jpg
                │   │   ├── 0002.jpg
                │   │   ├── 0003.jpg
                │   │   ├── 0004.jpg
                │   │   ├── 0005.jpg
                │   │   └── ...
                │   ├── Clip2/
                │   ├── Clip3/
                │   ├── ...
                │   └── Clip10/
                ├── game2/
                ├── game3/
                ├── ...
                └── game10/
      </pre>
    </details>


## 🏸 Badminton Dataset

- Introduced in [TrackNetV2](https://ieeexplore.ieee.org/document/9302757) paper.

### 📊 Evaluation Protocol

We follow the same evaluation protocol presented in the original paper. Although the dataset now contains 23 professional matches - likely due to a later update - the TrackNetV2 paper appears to use only 15. To ensure a fair comparison, we also use only these 15 professional matches.

### 🔧 Setup Instructions

1. Download the dataset using the Sharepoint link provided in this [website](https://hackmd.io/@TUIK/rJkRW54cU).
2. Unzip the `TrackNetV2.zip` file.
3. Move the extracted `Dataset` folder into the `data/badminton/` directory as shown below:

    <details>
      <summary>Expected Directory Structure (Before Preprocessing)</summary>
    
      <pre>
    repo-root
    └── data
        └── badminton
            └── TrackNetV2
                ├── Amateur
                │   ├── match1
                │   │   ├── csv
                │   │   │   ├── 1_00_01_ball.csv
                │   │   │   ├── 1_01_01_ball.csv
                │   │   │   ├── ...
                │   │   │   └── 1_05_05_ball.csv
                │   │   └── video
                │   │       ├── 1_00_01.mp4
                │   │       ├── 1_01_01.mp4
                │   │       ├── ...
                │   │       └── 1_05_05.mp4
                │   ├── match2
                │   └── match3
                ├── Professional
                └── Test
      </pre>
    </details>

4. Run the command below to preprocess the data:

   ```bash
   python3 scripts/preprocess.py --dataset badminton
   ```

    <details>
      <summary>Expected Directory Structure (After Preprocessing)</summary>
    
    <pre>
    repo-root/
    └── data/
      └── tennis/
          └── Dataset/
              ├── game1/
              │   ├── Clip1/
              │   │   ├── 0000.jpg
              │   │   ├── 0001.jpg
              │   │   ├── 0002.jpg
              │   │   ├── 0003.jpg
              │   │   ├── 0004.jpg
              │   │   ├── 0005.jpg
              │   │   └── ...
              │   ├── Clip2/
              │   ├── Clip3/
              │   ├── ...
              │   └── Clip10/
              ├── game2/
              ├── game3/
              ├── ...
              └── game10/
    </pre>
    </details>

> ℹ️ The default target resolution is 512×288 (width × height), and the default sequence length is 3 input frames → 3 output frames.
These settings were used consistently across all our experiments.


## 🚀🎾 New Tennis Dataset


Coming soon.


## ⚙️ Customising Dataset

A base dataset class is provided in [`src/dataset.py`](/src/dataset.py). You can inherit this class to create a custom dataset handler by implementing the `process_data` method.

The `process_data` method should process your raw dataset and generate output into `train` and `test` subfolders within the `processed_data` directory. Each of these folders must contain `.npy` files in the following format:

- `x_data_n.npy`  
- `y_data_n.npy`  

Where `n` is the index, starting from 1.

**Data Format:**

- **`x_data_n.npy`**: A NumPy array of shape `(num_frames_in_video - 2, 9, 288, 512)`, where:
  - `9` = 3 frames × 3 channels (e.g., RGB)
  - `288` is the frame height
  - `512` is the frame width

- **`y_data_n.npy`**: A corresponding NumPy array of shape `(num_frames_in_video - 2, 3, 288, 512)` containing heatmaps of the ball:
  - `3` = 3 heatmaps, one for each frame
  - `288` is the frame height
  - `512` is the frame width

📌 *Refer to the examples in [`src/dataset.py`](/src/dataset.py) for guidance on how to implement the `process_data` method.*

