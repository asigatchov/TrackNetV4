#!/usr/bin/env python
"""
Data Preprocessing Script
----------------------------------------
This script preprocesses data for the TrackNet model.

Usage:
    python preprocess_data.py --dataset <dataset_name> --height <height> --width <width>

Example:
    python preprocess_data.py --dataset tennis_game_level_split --height 288 --width 512

Arguments:
    --dataset      : Name of the dataset to use.
                     Allowed values: tennis_game_level_split, tennis_clip_level_split, badminton, new_tennis.
    --height     : Target height for the images.
    --width      : Target width for the images.

"""

from util import get_dataset
import argparse

def preprocess_dataset(dataset_name, height, width):
    """Preprocess specified dataset using provided target image dimensions."""
    dataset = get_dataset(dataset_name, "train", height, width)
    dataset.process_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess dataset images by resizing them to the specified height and width."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['tennis_game_level_split', 'tennis_clip_level_split', 'badminton', 'new_tennis'],
        help="Name of the dataset to use."
    )
    parser.add_argument('--height', type=int, default=288, help="Target image height (default: 288).")
    parser.add_argument('--width', type=int, default=512, help="Target image width (default: 512).")

    args = parser.parse_args()
    print("Preprocessing Configurations:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    preprocess_dataset(args.dataset, args.height, args.width)
