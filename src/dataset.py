import os
import json
import numpy as np
import cv2
import pandas as pd
from glob import glob
from collections import defaultdict
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import shutil

from util import genHeatMap
from constants import (
    SIGMA,
    MAG,
    WIDTH,
    HEIGHT,
    TENNIS_DATASET_GAME_LEVEL_SPLIT_CSV,
    TENNIS_DATASET_CLIP_LEVEL_SPLIT_CSV,
)


class BaseDataset():
    def __init__(self, root_dir, mode, target_img_height=HEIGHT, target_img_width=WIDTH, sequence_dim=(3,3), mag=MAG, sigma=SIGMA, shuffle=True):
        self.root_dir = root_dir
        self.mode = mode
        self.target_img_height = target_img_height
        self.target_img_width = target_img_width
        self.sequence_dim = sequence_dim
        self.processed_folder = os.path.join(root_dir, "processed_data", mode)
        self.shuffle = shuffle
        self.mag = mag
        self.sigma = sigma
        self.data_files = []
        self._refresh_file_list()

    def _refresh_file_list(self):
        """
        Refresh the list of processed data files in the processed_folder.
        """
        if os.path.exists(self.processed_folder):
            self.data_files = sorted(glob(os.path.join(self.processed_folder, "*.npz")))
        else:
            self.data_files = []

    def __len__(self):
        """
        Returns the number of processed sample pairs (x and y).
        """
        self._refresh_file_list()
        return len(self.data_files)

    def _getitem(self, idx):
        """
        Loads x and y data for the given index.
        """
        #self._refresh_file_list()
        data_path = self.data_files[idx]
        file_size = os.path.getsize(data_path)/1000/1000
        print(f"Loading data from {data_path}, size: {file_size} Mbytes")
        data = np.load(data_path)
        x = data['x']
        y = data['y']
        del data
        return x, y

    def _is_processed(self):
        """
        Checks if the processed .npy files exist for the current mode.
        """
        print(f"Checking if dataset is processed in {self.processed_folder}...")
        if not os.path.exists(self.processed_folder):
            print(f"Processed folder does not exist: {self.processed_folder}")
            return False

        npz_files = glob(os.path.join(self.processed_folder, "*.npz"))
        if npz_files:
            return True
        else:
            print(f"No .npz files found in {self.processed_folder}")
            return False

    def process_data(self):
        """
        Processes the dataset and creates processed .npz files.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the process_data method.")

    def __iter__(self):
        """
        Generator to iterate over dataset sample file names.
        """
        if not self._is_processed():
            print("Dataset not processed yet. Please run the `process_data` method first.")
            return

        self._refresh_file_list()
        indices = list(range(len(self.data_files)))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in indices:
            yield self._getitem(i)


class BadmintonDataset(BaseDataset):
    def __init__(self, root_dir, mode, target_img_height=HEIGHT, target_img_width=WIDTH, sequence_dim=(3,3), mag=MAG, sigma=SIGMA, shuffle=True):
        super().__init__(root_dir, mode, target_img_height, target_img_width, sequence_dim, mag, sigma, shuffle)

    def process_data(self):
        """
        Processes the badminton dataset by handling different match groups.
        """
        train_save_dir = os.path.join(self.root_dir, "processed_data", "train")
        test_save_dir = os.path.join(self.root_dir, "processed_data", "test")

        # Process Professional and Amateur matches for training data.
        train_count = self._process_helper(
            subfolder="Professional",
            match_list=[
                'match1',
                #'match2',
#                'match3', 'match4', 'match5', 'match6',
#                'match7', 'match8', 'match9', 'match10', 'match11', 'match12',
#                'match13', 'match14', 'match15'
            ],
            save_dir=train_save_dir,
            file_count=1
        )
        self._process_helper(
            subfolder="Amateur",
            match_list=['match1'] , #, 'match2', 'match3'],
            save_dir=train_save_dir,
            file_count=train_count
        )
        # Process Test matches.
        self._process_helper(
            subfolder="Test",
            match_list=['match1' ], # , 'match2', 'match3'],
            save_dir=test_save_dir,
            file_count=1
        )

    def _process_helper(self, subfolder, match_list, save_dir, file_count):
        """
        Helper function to process a list of matches within a specified subfolder.

        Args:
            subfolder (str): The subfolder within the dataset (e.g., "Professional").
            match_list (list): List of match folder names.
            save_dir (str): Directory name ("train" or "test") to save processed data.
            file_count (int): Starting count for naming output files.

        Returns:
            int: Updated file_count after processing.
        """
        os.makedirs(save_dir, exist_ok=True)

        for match in match_list:
            match_folder = os.path.join(self.root_dir, "TrackNetV2", subfolder, match)
            video_pattern = os.path.join(match_folder, 'video', '*.mp4')
            video_list = glob(video_pattern)

            # Create output directory for extracted frames.
            frames_dir = os.path.join(match_folder, 'frame')
            os.makedirs(frames_dir, exist_ok=True)

            # Extract frames from each video.
            for video_path in video_list:
                rally_name = os.path.basename(video_path).replace('.mp4', '')
                output_path = os.path.join(frames_dir, rally_name)
                os.makedirs(output_path, exist_ok=True)

                cap = cv2.VideoCapture(video_path)
                success, count = True, 0
                while success:
                    success, frame = cap.read()
                    if success:
                        cv2.imwrite(os.path.join(output_path, f'{count}.png'), frame)
                        count += 1
                cap.release()

            # Process CSV labels and generate training/test samples.
            rally_dirs = glob(os.path.join(match_folder, 'frame', '*'))
            rally_names = [os.path.basename(rally_dir) for rally_dir in rally_dirs]

            for rally in rally_names:
                label_path = os.path.join(match_folder, 'csv', f'{rally}_ball.csv')
                data = pd.read_csv(label_path)
                frames_numbers = data['Frame'].values
                visibilities = data['Visibility'].values
                x_coords = data['X'].values
                y_coords = data['Y'].values
                num_frames = len(frames_numbers)
                rally_path = os.path.join(match_folder, 'frame', rally)

                x_data_list = []
                y_data_list = []

                # Compute resize ratio using a sample image.
                sample_img = img_to_array(load_img(os.path.join(rally_path, "0.png")))
                ratio = sample_img.shape[0] / self.target_img_height

                # Process sequences of 3 consecutive frames.
                for i in range(num_frames - (self.sequence_dim[0] - 1)):
                    # Process x data (image sequences)
                    frames_sequence = []
                    for j in range(self.sequence_dim[0]):
                        frame_filename = f"{frames_numbers[i + j]}.png"
                        frame_path = os.path.join(rally_path, frame_filename)
                        img = load_img(frame_path, target_size=(self.target_img_height, self.target_img_width))
                        img_array = img_to_array(img)
                        img_array = np.moveaxis(img_array, -1, 0)
                        frames_sequence.extend([img_array[0], img_array[1], img_array[2]])
                    x_data_list.append(np.stack(frames_sequence, axis=0))

                    # Process y data (generate heatmaps)
                    heatmap_sequence = []
                    for j in range(self.sequence_dim[1]):
                        if visibilities[i + j] == 0:
                            heatmap = genHeatMap(self.target_img_width, self.target_img_height, -1, -1, self.sigma, self.mag)
                        else:
                            heatmap = genHeatMap(self.target_img_width, self.target_img_height, int(x_coords[i + j] / ratio),
                                                int(y_coords[i + j] / ratio), self.sigma, self.mag)
                        heatmap_sequence.append(heatmap)
                    y_data_list.append(heatmap_sequence)

                # Convert collected data to NumPy arrays.
                x_data = np.asarray(x_data_list, dtype='float32') / 255.0
                y_data = np.asarray(y_data_list)

                print('x_data:', x_data.shape)
                print('y_data', y_data.shape)

                # Save the processed data.

                np.savez_compressed(
                    os.path.join(save_dir, f"data_{file_count}.npz"), x=x_data, y=y_data
                )
                file_count += 1

            # Delete the frames directory after processing the match.
            shutil.rmtree(frames_dir)

        return file_count


class TennisDataset(BaseDataset):
    """
    TennisDataset processes tennis video data for training track networks.

    References:
      - Paper: https://ieeexplore.ieee.org/document/8909871
      - Dataset: https://github.com/yastrebksv/TrackNet (retrieve the provided Dataset.zip file).

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): "game_level" or "clip_level" split.
        mode (str): Dataset mode (e.g., "train", "test").
        target_img_size (tuple): Tuple (WIDTH, HEIGHT) for resizing images.
        sequence_dim (tuple): Dimensions for the sequence (e.g., (3, 3)).
        mag (float): Magnification parameter for heatmap generation.
        sigma (float): Sigma value for Gaussian heatmaps.
        shuffle (bool): Whether to shuffle the dataset.
    """
    def __init__(self, root_dir, split, mode, target_img_height=HEIGHT, target_img_width=WIDTH, sequence_dim=(3, 3), mag=MAG, sigma=SIGMA, shuffle=True):
        super().__init__(root_dir, mode, target_img_height, target_img_width, sequence_dim, mag, sigma, shuffle)
        if split not in ["game_level", "clip_level"]:
            raise ValueError("Unknown split name passed to TennisDataset class")

        self.split = split
        self.processed_folder = os.path.join(root_dir, "processed_data", self.split, self.mode)

    @staticmethod
    def get_clip_level_split(csv_path_or_df):
        """
        Given a CSV path or DataFrame with columns 'game', 'clip', and 'set',
        returns two dictionaries: train_dict and test_dict.
        Each dictionary maps a game to a list of its associated clips.
        """
        df = pd.read_csv(csv_path_or_df) if isinstance(csv_path_or_df, str) else csv_path_or_df

        train_dict = defaultdict(list)
        test_dict = defaultdict(list)

        for _, row in df.iterrows():
            if row['set'] == 'train':
                train_dict[row['game']].append(row['clip'])
            elif row['set'] == 'test':
                test_dict[row['game']].append(row['clip'])

        return dict(train_dict), dict(test_dict)

    @staticmethod
    def get_game_level_split(csv_path):
        """
        Reads a CSV with columns 'game' and 'set', and returns two lists:
        train_games and test_games.
        """
        df = pd.read_csv(csv_path)
        train_games = df[df['set'] == 'train']['game'].tolist()
        test_games = df[df['set'] == 'test']['game'].tolist()
        return train_games, test_games

    def process_data(self):
        if self.split == "game_level":
            self._process_game_level()
        elif self.split == "clip_level":
            self._process_clip_level()
        print("Processing completed.")

    def _process_clip_level(self):
        train_set, test_set = self.get_clip_level_split(TENNIS_DATASET_GAME_LEVEL_SPLIT_CSV)
        self._process_set(train_set, os.path.join(self.root_dir, "processed_data", "clip_level", "train"))
        self._process_set(test_set, os.path.join(self.root_dir, "processed_data", "clip_level", "test"))

    def _process_game_level(self):
        train_games, test_games = self.get_game_level_split(TENNIS_DATASET_CLIP_LEVEL_SPLIT_CSV)
        self._process_set(train_games, os.path.join(self.root_dir, "processed_data", "game_level", "train"), use_clip_list=False)
        self._process_set(test_games, os.path.join(self.root_dir, "processed_data", "game_level", "test"), use_clip_list=False)

    def _process_set(self, set_data, save_data_dir, use_clip_list=True):
        """
        Process a set of clips or games.

        :param set_data: Either a dictionary (game->list of clips) or a list of game names.
        :param save_data_dir: Directory to save processed data.
        :param use_clip_list: If True, set_data is a dict of game:clips; if False, process all clips in each game folder.
        """
        os.makedirs(save_data_dir, exist_ok=True)
        count = 1

        if use_clip_list:
            # set_data is a dict: game -> list of clips
            for game, clips in set_data.items():
                game_folder = os.path.join(self.root_dir, "Dataset", game)
                for clip in clips:
                    clip_folder = os.path.join(game_folder, clip)
                    print(f"Processing game: {game}, clip: {clip}")
                    x_data, y_data = self._process_clip(clip_folder)
                    self._save_data(save_data_dir, count, x_data, y_data)
                    count += 1
        else:
            # set_data is a list of games; process all clips in each game folder.
            for game in set_data:
                game_folder = os.path.join(self.root_dir, "Dataset", game)
                # list directories inside game_folder as clips
                clips = [d for d in os.listdir(game_folder) if os.path.isdir(os.path.join(game_folder, d))]
                for clip in clips:
                    clip_folder = os.path.join(game_folder, clip)
                    print(f"Processing game: {game}, clip: {clip}")
                    x_data, y_data = self._process_clip(clip_folder)
                    self._save_data(save_data_dir, count, x_data, y_data)
                    count += 1

    def _process_clip(self, clip_folder):
        """
        Process a single clip folder. Reads Label.csv and processes each frame sequence.

        Returns:
            x_data: Processed numpy array of shape (B, C, T, H, W)
            y_data: Processed numpy array of heatmaps.
        """
        label_path = os.path.join(clip_folder, 'Label.csv')
        label_data = pd.read_csv(label_path)
        file_names = label_data['file name'].values
        visibility = label_data['visibility'].values
        x_coords = label_data['x-coordinate'].values
        y_coords = label_data['y-coordinate'].values

        num_frames = file_names.shape[0]

        sample_image = img_to_array(load_img(os.path.join(clip_folder, file_names[0])))
        ratio = sample_image.shape[0] / self.target_img_height

        x_data_list = []
        y_data_list = []

        # Process sequences of 3 consecutive frames.
        for i in range(num_frames - (self.sequence_dim[0] - 1)):
            # Process x data (image sequences)
            frames_sequence = []
            for j in range(self.sequence_dim[0]):
                frame_path = os.path.join(clip_folder, str(file_names[i + j]))
                img = load_img(frame_path, target_size=(self.target_img_height, self.target_img_width))
                img_array = img_to_array(img)
                img_array = np.moveaxis(img_array, -1, 0)
                frames_sequence.extend([img_array[0], img_array[1], img_array[2]])
            x_data_list.append(np.stack(frames_sequence, axis=0))

            # Process y data (generate heatmaps)
            heatmap_sequence = []
            for j in range(self.sequence_dim[1]):
                if visibility[i + j] == 0:
                    heatmap = genHeatMap(self.target_img_width, self.target_img_height, -1, -1, self.sigma, self.mag)
                else:
                    heatmap = genHeatMap(self.target_img_width, self.target_img_height, int(x_coords[i + j] / ratio),
                                        int(y_coords[i + j] / ratio), self.sigma, self.mag)
                heatmap_sequence.append(heatmap)
            y_data_list.append(heatmap_sequence)

        # Convert collected data to NumPy arrays.
        x_data = np.asarray(x_data_list, dtype='float32') / 255.0
        y_data = np.asarray(y_data_list)

        return x_data, y_data

    def _save_data(self, save_dir, count, x_data, y_data):
        """
        Saves processed data as npy files.
        """
        np.save(os.path.join(save_dir, f'x_data_{count}.npy'), x_data)
        np.save(os.path.join(save_dir, f'y_data_{count}.npy'), y_data)


class NewTennisDataset(BaseDataset):
    def process_data(self):
        pass
