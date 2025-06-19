from util import get_dataset
import argparse

def preprocess_dataset(dataset_name, height, width, chunk_size=1000):
    """Preprocess specified dataset using provided target image dimensions and chunk size."""
    dataset = get_dataset(dataset_name, "train", height, width, chunk_size)
    dataset.process_data(chunk_size=chunk_size)

    dataset = get_dataset(dataset_name, "test", height, width, chunk_size)
    dataset.process_data(chunk_size=chunk_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess dataset images by resizing them to the specified height and width."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['tennis_game_level_split', 'tennis_clip_level_split', 'badminton', 'new_tennis', 'custom_dataset'],
        help="Name of the dataset to use."
    )
    parser.add_argument('--height', type=int, default=288, help="Target image height (default: 288).")
    parser.add_argument('--width', type=int, default=512, help="Target image width (default: 512).")
    parser.add_argument('--chunk_size', type=int, default=1000, help="Number of sequences per chunk for HDF5 storage (default: 1000).")

    args = parser.parse_args()
    print("Preprocessing Configurations:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    preprocess_dataset(args.dataset, args.height, args.width, args.chunk_size)