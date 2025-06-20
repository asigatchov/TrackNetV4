import os
import numpy as np
import cv2
import h5py
from glob import glob
from util import genHeatMap
from constants import SIGMA, MAG, WIDTH, HEIGHT

def visualize_hdf5_dataset(dataset_dir=None, mode='train', data_file=None, sequence_length=3):
    """
    Visualize frames and object annotations from HDF5 dataset using OpenCV.
    Press 'n' to move to the next frame, 'p' to move to the previous frame, 'q' to quit.

    Args:
        dataset_dir (str): Root directory of the dataset (e.g., 'datasetdir/').
        mode (str): Dataset mode ('train' or 'test').
        data_file (str): Path to a specific HDF5 file to visualize (optional).
        sequence_length (int): Number of frames in each sequence.
    """
    if data_file is not None:
        hdf5_files = [data_file]
    else:
        if dataset_dir is None:
            print("Either --data_file or --dataset_dir must be specified")
            return
        processed_folder = os.path.join(dataset_dir, "processed_data", mode)
        hdf5_files = sorted(glob(os.path.join(processed_folder, "*.h5")))

    if not hdf5_files:
        print("No HDF5 files found.")
        return

    for h5_path in hdf5_files:
        print(f"Opening {h5_path}")
        with h5py.File(h5_path, "r", swmr=True, libver="latest") as f:
            num_samples = f["x"].shape[0]
            x_shape = f["x"].shape
            y_shape = f["y"].shape
            print(f"Samples: {num_samples}, x shape: {x_shape}, y shape: {y_shape}")

            idx = 0
            while True:
                # Чтение одного сэмпла покадрово
                frames = []
                for i in range(sequence_length):
                    print(f"Reading sample {idx+1}/{num_samples}, frame {i+1}/{sequence_length}")
                    frame_arr = f["x"][idx, i*3:(i+1)*3, :, :]
                    frame = np.moveaxis(frame_arr, 0, -1)
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                    frames.append(frame)

                for i, frame in enumerate(frames):
                    heatmap = f["y"][idx, i, :, :]
                    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                    heatmap_uint8 = heatmap_norm.astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)
                    if np.max(heatmap_uint8) > 10:
                        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(heatmap_uint8)
                        cv2.circle(overlay, maxLoc, 5, (0,255,0), 2)
                    cv2.putText(overlay, f"Sample {idx+1}/{num_samples} Frame {i+1}/{sequence_length}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.namedWindow(
                        "Frame+Heatmap", cv2.WINDOW_NORMAL
                    )  # Create window with freedom of dimensions

                    cv2.imshow("Frame+Heatmap", overlay)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('n'):
                        continue
                    elif key == ord('p'):
                        idx = max(idx-1, 0)
                        break
                else:
                    idx += 1
                if idx >= num_samples:
                    print("End of file.")
                    break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize HDF5 dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to your dataset directory",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Dataset mode: train or test",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Path to a specific HDF5 file to visualize",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=3,
        help="Number of frames in each sequence",
    )
    args = parser.parse_args()
    visualize_hdf5_dataset(
        args.dataset_dir,
        mode=args.mode,
        data_file=args.data_file,
        sequence_length=args.sequence_length,
    )
