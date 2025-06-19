#!/usr/bin/env python
"""
Training Script
-------------------------

This script trains a TrackNet model on a specified dataset using configurable parameters.
It supports loading a pretrained model, saving checkpoints, and evaluating performance during training.

Usage:
    python src/train.py --model_name <MODEL> --dataset <DATASET> [options]

Example:
    python src/train.py --model_name Baseline_TrackNetV2 --dataset tennis_game_level_split \
        --batch_size 2 --learning_rate 1.0 --height 288 --width 512 --epochs 30 --tol 4 \
        --work_dir ./models --save_freq 1

Arguments:
    --model_name   : Name of the model to use.
                     Allowed values: Baseline_TrackNetV2, TrackNetV4_TypeA, TrackNetV4_TypeB.
    --dataset      : Name of the dataset to use.
                     Allowed values: tennis_game_level_split, tennis_clip_level_split, badminton, new_tennis.
    --batch_size   : Batch size for training (default: 2).
    --learning_rate: Learning rate for the optimizer (default: 1.0).
    --height       : Target image height (default: 288).
    --width        : Target image width (default: 512).
    --epochs       : Number of epochs for training (default: 30).
    --tol          : Tolerance for the outcome evaluation (default: 4).
    --model_path   : Path to a pretrained model (.keras) to load before training (optional).
    --work_dir     : Directory to save the trained models (default: "./models").
    --save_freq    : Frequency (in epochs) to save model checkpoints (default: 1).

Note:
    If the default work directory is used, a timestamp will be appended to create a unique directory.
"""

import argparse
import datetime
import os
import io
import tempfile

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from util import custom_loss, get_dataset, get_model, outcome
from models.TrackNetV4 import (
    #MotionPromptLayer,
    FusionLayerTypeA,
    FusionLayerTypeB
)

from models.TrackNetV4Fast import MotionPromptLayer


def main(args):
    """
    Train the TrackNet model using specified configurations.
    """
    # Unpack arguments
    model_name = args.model_name
    dataset_name = args.dataset
    batch_size = args.batch_size
    height = args.height
    width = args.width
    learning_rate = args.learning_rate
    epochs = args.epochs
    tol = args.tol
    save_freq = args.save_freq
    model_path = args.model_path
    work_dir = args.work_dir

    # If using default work directory, append a timestamp for uniqueness
    if work_dir == "./models":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = os.path.join(work_dir, timestamp)

    # Print all experiment configurations before starting the training
    experiment_config = {
        "model_name": model_name,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "height": height,
        "width": width,
        "epochs": epochs,
        "tol": tol,
        "model_path": model_path,
        "work_dir": work_dir,
        "save_freq": save_freq,
    }
    print("Training Configurations:")
    for key, value in experiment_config.items():
        print(f"  {key}: {value}")

    # Create the work directory if it doesn't exist
    os.makedirs(work_dir, exist_ok=True)

    # Load model architecture
    model = get_model(model_name, height, width)

    # Optionally load pretrained weights
    if model_path:
        custom_objects = {
            'custom_loss': custom_loss,
            # Following for TrackNetV4
            'MotionPromptLayer': MotionPromptLayer,
            'FusionLayerTypeA': FusionLayerTypeA,
            'FusionLayerTypeB': FusionLayerTypeB,
        }
        # This block ensures model weights can be loaded even if there are architectural differences
        # (i.e. additional layers like fusion layers) between the saved model and the current one.
        # By saving weights from the loaded model and then reloading them into the current model
        # using `skip_mismatch=True`, it avoids errors caused by shape or structure mismatches.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model = load_model(model_path, custom_objects=custom_objects)
            model_temp_path = os.path.join(temp_dir, "temp_weights.weights.h5")
            print(f"Loading pretrained model from {model_path}...")
            print(f"Saving temporary model weights to {model_temp_path}...")
            temp_model.save_weights(model_temp_path)
            model.load_weights(model_temp_path, skip_mismatch=True)


    # Load training dataset
    dataset_train = get_dataset(dataset_name, "train")

    # Compile the model with Adadelta optimizer and custom loss
    model.compile(
        loss=custom_loss,
        optimizer=Adadelta(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    callbacks = [
        TensorBoard(log_dir='./logs', histogram_freq=1),  # Логи для TensorBoard
        EarlyStopping(monitor='loss', patience=3, verbose=1),  # Остановка при стагнации
        ModelCheckpoint('best_model.keras', save_best_only=True, verbose=1)  # Сохранение лучшей модели
    ]
    # Main training loop
    for epoch in range(epochs):
        print(f"======== Epoch {epoch + 1} ========")

        # Train on each batch in the training dataset
        len_dataset = len(dataset_train)
        
        cnt = 0
        for x_train, y_train in dataset_train:
            cnt += 1
            print(f"Processing batch {cnt}/{len_dataset}...")
            print(x_train.shape, y_train.shape)
            print(x_train.dtype, y_train.dtype)
            print('#' * 50)
            model.fit(x_train, y_train, batch_size=batch_size, epochs=1, callbacks=callbacks, verbose=0)
            del x_train, y_train


        # if (epoch + 1) % 5 == 0:
        #     print(f"Epoch {epoch + 1} completed. Run predictions")
        #     # Evaluate model performance on the training set
        #     TP = TN = FP1 = FP2 = FN = 0
        #     for x_train, y_train in dataset_train:
        #         y_pred = model.predict(x_train, batch_size=batch_size)
        #         y_pred = (y_pred > 0.5).astype('float32')

        #         tp, tn, fp1, fp2, fn = outcome(y_pred, y_train, tol)
        #         TP += tp
        #         TN += tn
        #         FP1 += fp1
        #         FP2 += fp2
        #         FN += fn

        #         del x_train, y_train, y_pred

        #     print(f"Epoch {epoch + 1} results: TP={TP}, TN={TN}, FP1={FP1}, FP2={FP2}, FN={FN}")
        # else:
        #     print(f"Epoch {epoch + 1} completed. No evaluation this time.")

        # Save model checkpoint based on frequency
        if  True or (epoch + 1) % save_freq == 0:
            model_path = os.path.join(work_dir, f"model_{epoch + 1}.keras")
            model.save(model_path)
            print(f"Saved model to {model_path}")

    # Save the final model after training
    final_model_path = os.path.join(work_dir, "model_final.keras")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TrackNet model with configurable parameters."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
       # choices=['Baseline_TrackNetV2', 'TrackNetV4_TypeA', 'TrackNetV4_TypeB', 'TrackNetV4_Nano', 'TrackNetV4_Small'],
        help="Name of the model to use."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['tennis_game_level_split', 'tennis_clip_level_split', 'badminton', 'new_tennis', 'custom_dataset'],
        help="Name of the dataset to use."
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate for the optimizer")
    parser.add_argument("--height", type=int, default=288, help="Target height of the images")
    parser.add_argument("--width", type=int, default=512, help="Target width of the images")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--tol", type=int, default=4, help="Tolerance for the outcome evaluation")

    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model (.keras) to load before training"
    )
    parser.add_argument("--work_dir", type=str, default="./models", help="Directory to save the trained models")
    parser.add_argument("--save_freq", type=int, default=1, help="Frequency (in epochs) to save model checkpoints")


    args = parser.parse_args()
    main(args)
