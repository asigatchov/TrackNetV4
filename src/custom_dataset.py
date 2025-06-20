import os
import json
import numpy as np
import cv2
import pandas as pd
from glob import glob
from collections import deque
import h5py
import threading
import time
import gc
from util import genHeatMap
from constants import SIGMA, MAG, WIDTH, HEIGHT
from multiprocessing import Pool, cpu_count

class BaseCustomDataset:
    def __init__(self, root_dir, mode, target_img_height=HEIGHT, target_img_width=WIDTH, sequence_dim=(3,3), mag=MAG, sigma=SIGMA, shuffle=True, buffer_size=5):
        """
        Initialize the base dataset for custom data.

        Args:
            root_dir (str): Root directory containing data (e.g., 'datasetdir/').
            mode (str): Dataset mode ('train' or 'test').
            target_img_height (int): Target height for resized frames.
            target_img_width (int): Target width for resized frames.
            sequence_dimUber (tuple): Dimensions for input and output sequences (e.g., (3,3)).
            mag (float): Magnification factor for heatmaps.
            sigma (float): Sigma for heatmap generation.
            shuffle (bool): Whether to shuffle data during iteration.
            buffer_size (int): Number of data chunks to preload in memory.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.target_img_height = target_img_height
        self.target_img_width = target_img_width
        self.sequence_dim = sequence_dim
        self.processed_folder = os.path.join(root_dir, "processed_data", mode)
        self.metadata_file = os.path.join(self.processed_folder, "metadata.json")
        self.shuffle = shuffle
        self.mag = mag
        self.sigma = sigma
        self.data_files = []
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        self.stop_thread = threading.Event()
        self._refresh_file_list()
        self._start_loader_thread()

    def _refresh_file_list(self):
        """Refresh the list of processed HDF5 files."""
        os.makedirs(self.processed_folder, exist_ok=True)
        self.data_files = sorted(glob(os.path.join(self.processed_folder, "*.h5")))

    def __len__(self):
        """Return the number of processed HDF5 files."""
        self._refresh_file_list()
        return len(self.data_files)

    def _load_data(self, data_path, chunk_size=240):
        """Yield x and y data chunks from an HDF5 file, up to 240 sequences each."""
        print(f"Loading data from {data_path}, size: {os.path.getsize(data_path) / 1000 / 1000:.2f} MB")
        start = time.time()
        with h5py.File(data_path, 'r', swmr=True, libver='latest') as f:
            x_dataset = f['x']
            y_dataset = f['y']
            total_samples = x_dataset.shape[0]
            for i in range(0, total_samples, chunk_size):
                x_chunk = x_dataset[i:i + chunk_size]
                y_chunk = y_dataset[i:i + chunk_size]
                print(f"Yielding chunk: x={x_chunk.shape}, y={y_chunk.shape}")
                yield (x_chunk, y_chunk)
                gc.collect()  # Free memory after yielding each chunk
        print(f"Finished yielding chunks from {data_path} in {time.time() - start:.2f} seconds")

    def _loader_thread(self):
        """Background thread to preload data chunks into buffer from multiple files."""
        indices = list(range(len(self.data_files)))
        if self.shuffle:
            np.random.shuffle(indices)
        current_idx = 0
        while not self.stop_thread.is_set():
            # Check buffer size without holding lock to reduce contention
            if len(self.buffer) >= self.buffer_size:
                time.sleep(0.01)  # Reduced sleep time for faster response
                continue
            if current_idx >= len(self.data_files):
                if self.shuffle:
                    np.random.shuffle(indices)
                current_idx = 0
            data_path = self.data_files[indices[current_idx]]
            try:
                # Preload chunks from the current file
                for chunk in self._load_data(data_path):
                    with self.buffer_lock:
                        if len(self.buffer) < self.buffer_size:
                            self.buffer.append(chunk)
                        else:
                            break  # Stop loading if buffer is full
                current_idx += 1
            except Exception as e:
                print(f"Error loading {data_path}: {e}")
                current_idx += 1
                continue
            # Try to preload from the next file if buffer isn't full
            with self.buffer_lock:
                if len(self.buffer) < self.buffer_size and current_idx < len(self.data_files):
                    next_idx = (current_idx + 1) % len(self.data_files)
                    next_path = self.data_files[indices[next_idx]]
                    try:
                        for chunk in self._load_data(next_path):
                            if len(self.buffer) < self.buffer_size:
                                self.buffer.append(chunk)
                            else:
                                break
                    except Exception as e:
                        print(f"Error preloading {next_path}: {e}")

    def _start_loader_thread(self):
        """Start the background loader thread if processed data exists."""
        if self._is_processed():
            self.loader_thread = threading.Thread(target=self._loader_thread, daemon=True)
            self.loader_thread.start()

    def _getitem(self, idx):
        """Fetch a chunk from buffer or load directly."""
        with self.buffer_lock:
            if len(self.buffer) > 0:
                return self.buffer.popleft()
        print("Buffer empty, loading directly")
        chunks = self._load_data(self.data_files[idx])
        return next(chunks, None)  # Return first chunk or None if no chunks

    def _is_processed(self):
        """Check if processed HDF5 files exist."""
        print(f"Checking processed data in {self.processed_folder}...")
        return len(self.data_files) > 0

    def __iter__(self):
        """Generator to iterate over dataset samples."""
        if not self._is_processed():
            print("Dataset not processed. Run `process_data` first.")
            return
        self._refresh_file_list()
        indices = list(range(len(self.data_files)))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in indices:
            for chunk in self._load_data(self.data_files[i]):
                yield chunk

    def __del__(self):
        """Stop the loader thread when the dataset object is destroyed."""
        self.stop_thread.set()
        if hasattr(self, 'loader_thread'):
            self.loader_thread.join()

    def add_data(self, video_path, csv_path):
        """Add a new video and CSV pair to the dataset."""
        self._process_single_video(video_path, csv_path, self.processed_folder)

    def remove_data(self, video_name):
        """Remove processed data for a specific video."""
        metadata = self._load_metadata()
        hdf5_file = metadata.get(video_name)
        if hdf5_file and os.path.exists(hdf5_file):
            os.remove(hdf5_file)
            del metadata[video_name]
            self._save_metadata(metadata)
            print(f"Removed data for {video_name}")
        else:
            print(f"No processed data found for {video_name}")
        self._refresh_file_list()

    def _load_metadata(self):
        """Load metadata from JSON file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata):
        """Save metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

# Move worker to module level to make it picklable
def process_video_worker(args):
    video_path, csv_path, processed_folder, video_name, hdf5_path = args
    import cv2
    import pandas as pd
    import numpy as np
    import gc
    from util import genHeatMap
    from constants import SIGMA, MAG, WIDTH, HEIGHT

    print(f"Processing video: {video_name}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return video_name, None
    data = pd.read_csv(csv_path)
    frames_numbers = data['Frame'].values
    visibilities = data['Visibility'].values
    x_coords = data['X'].values
    y_coords = data['Y'].values
    num_frames = len(frames_numbers)

    ret, sample_frame = cap.read()
    if not ret:
        print(f"Failed to read video: {video_path}")
        cap.release()
        return video_name, None
    target_img_height = HEIGHT
    target_img_width = WIDTH
    sigma = SIGMA
    mag = MAG
    sequence_dim = (3, 3)
    ratio = sample_frame.shape[0] / target_img_height
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    x_data_list = []
    y_data_list = []

    for i in range(0, num_frames - (sequence_dim[0] - 1)):
        frames_sequence = []
        for j in range(sequence_dim[0]):
            frame_idx = frames_numbers[i + j]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx} from {video_name}")
                break
            frame = cv2.resize(frame, (target_img_width, target_img_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_array = np.array(frame)
            img_array = np.moveaxis(img_array, -1, 0)
            frames_sequence.extend([img_array[0], img_array[1], img_array[2]])
        if len(frames_sequence) != sequence_dim[0] * 3:
            continue
        x_data_list.append(np.stack(frames_sequence, axis=0))

        heatmap_sequence = []
        for j in range(sequence_dim[1]):
            if visibilities[i + j] == 0:
                heatmap = genHeatMap(target_img_width, target_img_height, -1, -1, sigma, mag)
            else:
                heatmap = genHeatMap(target_img_width, target_img_height,
                                     int(x_coords[i + j] / ratio), int(y_coords[i + j] / ratio),
                                     sigma, mag)
            heatmap_sequence.append(heatmap)
        y_data_list.append(heatmap_sequence)

    cap.release()
    x_data = np.asarray(x_data_list, dtype='float32') / 255.0
    y_data = np.asarray(y_data_list)
    print(f"Saving video: x={x_data.shape}, y={y_data.shape} to {hdf5_path}")
    import h5py
    with h5py.File(hdf5_path, "w", swmr=True, libver="latest") as f:
        f.create_dataset('x', data=x_data, compression='gzip', chunks=(min(100, x_data.shape[0]), *x_data.shape[1:]))
        f.create_dataset('y', data=y_data, compression='gzip', chunks=(min(100, y_data.shape[0]), *y_data.shape[1:]))
    del x_data, y_data
    gc.collect()
    return video_name, hdf5_path

class CustomDataset(BaseCustomDataset):
    def process_data(self, chunk_size=1000):
        """Process the custom dataset incrementally (parallelized by video)."""
        metadata = self._load_metadata()
        video_pattern = os.path.join(self.root_dir, self.mode, "*", "video", "*.mp4")
        video_list = glob(video_pattern)
        tasks = []
        for video_path in video_list:
            video_name = os.path.basename(video_path).replace('.mp4', '')
            if video_name in metadata:
                print(f"Skipping already processed video: {video_name}")
                continue
            csv_path = os.path.join(os.path.dirname(os.path.dirname(video_path)), "csv", f"{video_name}_ball.csv")
            if not os.path.exists(csv_path):
                print(f"CSV file not found for {video_name}, skipping")
                continue
            hdf5_path = os.path.join(self.processed_folder, f"{video_name}.h5")
            tasks.append((video_path, csv_path, self.processed_folder, video_name, hdf5_path))

        if tasks:
            with Pool(processes=min(4, len(tasks))) as pool:
                results = pool.map(process_video_worker, tasks)
            # Update metadata after all processes
            for video_name, hdf5_path in results:
                if hdf5_path is not None:
                    metadata[video_name] = hdf5_path
            self._save_metadata(metadata)
        self._refresh_file_list()
