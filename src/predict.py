"""
Script for predicting trajectories on video using a trained model.
It processes the video, performs inference using the loaded model, and outputs the predictions
as annotated video and CSV files with trajectory points.

Usage:
    python script.py --video_path <path_to_video> --model_weights <path_to_model_weights> --output_dir <output_directory> [--queue_length <queue_length>] [--show_frame]
"""

import os
import sys
import cv2
import numpy as np
import time
import queue
import argparse
from tensorflow.keras.models import load_model
import keras.backend as K
from models.TrackNetV4 import MotionPromptLayer, FusionLayerTypeA, FusionLayerTypeB
from constants import HEIGHT, WIDTH
from util import custom_loss

# Constants
BATCH_SIZE = 1
INPUT_HEIGHT = 288
INPUT_WIDTH = 512

def run_model_inference(model, frames):
    """
    Pre-processes the frames and runs inference on the model.
    
    Args:
        model: Loaded model for inference.
        frames (list): List of frames to run inference on.
    
    Returns:
        predictions: Model predictions for the frames.
        inference_time (float): Time taken for model inference.
    """
    input_batch = []
    
    # Preprocess the frames for model input
    for frame in frames:
        # Convert RGB to BGR (OpenCV format) and resize
        resized_frame = cv2.resize(frame[..., ::-1], (INPUT_WIDTH, INPUT_HEIGHT))
        # Transpose to (channels, height, width)
        frame_array = np.transpose(resized_frame, (2, 0, 1))
        input_batch.extend(frame_array[:3])

    # Prepare input for model prediction
    input_batch = np.asarray(input_batch).reshape((1, 9, INPUT_HEIGHT, INPUT_WIDTH)).astype('float32') / 255
    
    # Perform prediction
    inference_start_time = time.time()
    predictions = model.predict(input_batch, batch_size=BATCH_SIZE, verbose=1)
    inference_end_time = time.time()

    inference_time = inference_end_time - inference_start_time
    return predictions, inference_time

def post_process_predictions(predictions, frame1, frame2, frame3, frame_count, video_writer, csv_output_path, width_ratio, height_ratio, predicted_points_queue, show_frame):
    """
    Post-processes the predictions, annotates the video, saves results, and optionally displays the frame.
    
    Args:
        predictions: Predictions from the model.
        frame1, frame2, frame3: Original frames.
        frame_count: Current frame count in the video.
        video_writer: Video writer object for saving frames.
        csv_output_path: Path to the CSV file for saving results.
        width_ratio: Ratio to adjust the predicted points' width.
        height_ratio: Ratio to adjust the predicted points' height.
        predicted_points_queue: Deque storing the predicted points.
        show_frame (bool): Whether to display the processed frame.
    
    Returns:
        bool: True if processing should continue, False if user requests exit.
    """
    binary_predictions = (predictions > 0.5).astype('float32')
    binary_heatmaps = (binary_predictions[0] * 255).astype('uint8')

    for i, current_frame in enumerate([frame1, frame2, frame3]):
        if np.amax(binary_heatmaps[i]) <= 0:
            with open(csv_output_path, 'a') as csv_file:
                csv_file.write(f"{frame_count},0,0,0\n")
            video_writer.write(current_frame)
            if show_frame:
                cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
                cv2.imshow('Frame', current_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False
        else:
            contours, _ = cv2.findContours(binary_heatmaps[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
            largest_bounding_box = max(bounding_boxes, key=lambda r: r[2] * r[3])
            
            # Calculate predicted center
            predicted_x_center = int(width_ratio * (largest_bounding_box[0] + largest_bounding_box[2] / 2))
            predicted_y_center = int(height_ratio * (largest_bounding_box[1] + largest_bounding_box[3] / 2))

            # Update deque and draw trajectory
            predicted_points_queue.appendleft((predicted_x_center, predicted_y_center))
            predicted_points_queue.pop()

            frame_copy = np.copy(current_frame)
            for point in predicted_points_queue:
                if point is not None:
                    cv2.circle(frame_copy, point, 5, (0, 255, 0), 2)
            cv2.circle(frame_copy, (predicted_x_center, predicted_y_center), 5, (0, 0, 255), -1)
            video_writer.write(frame_copy)

            # Display frame if enabled
            if show_frame:
                cv2.imshow('Frame', frame_copy)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False

            # Write results to CSV
            with open(csv_output_path, 'a') as csv_file:
                csv_file.write(f"{frame_count},1,{predicted_x_center},{predicted_y_center}\n")
        
        frame_count += 1
    
    return True

def main(args):
    """
    Main function to handle video processing, model inference, and result saving.
    """
    video_path = args.video_path
    model_weights_path = args.model_weights
    output_dir = args.output_dir
    queue_length = args.queue_length
    show_frame = args.show_frame
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the predicted points queue with the specified length
    predicted_points_queue = queue.deque([None] * queue_length)

    # Load the trained model with custom objects
    model = load_model(
        model_weights_path, 
        custom_objects={
            'MotionPromptLayer': MotionPromptLayer,
            'custom_loss': custom_loss,
        }
    )

    # Read input video and set up output video settings
    video_capture = cv2.VideoCapture(video_path)
    success, frame1 = video_capture.read()
    success, frame2 = video_capture.read()
    success, frame3 = video_capture.read()

    height_ratio = frame1.shape[0] / INPUT_HEIGHT
    width_ratio = frame1.shape[1] / INPUT_WIDTH
    video_size = (int(INPUT_WIDTH * width_ratio), int(INPUT_HEIGHT * height_ratio))
    frames_per_second = int(video_capture.get(cv2.CAP_PROP_FPS))

    if video_path.endswith('avi'):
        video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    elif video_path.endswith('mp4'):
        video_codec = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        print('Error: Video format must be .avi or .mp4')
        sys.exit(1)

    # Output files
    output_video_path = os.path.join(output_dir, os.path.basename(video_path[:-4] + '_predict' + video_path[-4:]))
    csv_output_path = os.path.join(output_dir, os.path.basename(video_path[:-4] + '_predict.csv'))

    with open(csv_output_path, 'w') as csv_file:
        csv_file.write('Frame,Visibility,X,Y\n')
    video_writer = cv2.VideoWriter(output_video_path, video_codec, frames_per_second, video_size)

    # Main loop
    frame_count = 0
    total_inference_time = 0.0
    start_time = time.time()

    while success:
        frames = [frame1, frame2, frame3]
        predictions, inference_time = run_model_inference(model, frames)
        continue_processing = post_process_predictions(
            predictions, frame1, frame2, frame3, frame_count, video_writer, 
            csv_output_path, width_ratio, height_ratio, predicted_points_queue, show_frame
        )
        
        if not continue_processing:
            break

        # Accumulate the total inference time
        total_inference_time += inference_time
        frame_count += 3  # Increment by 3 since we process three frames at a time

        # Read next set of frames
        success, frame1 = video_capture.read()
        success, frame2 = video_capture.read()
        success, frame3 = video_capture.read()

    # Clean up resources
    video_writer.release()
    video_capture.release()
    if show_frame:
        cv2.destroyAllWindows()
    end_time = time.time()

    # Output timing information
    if total_inference_time > 0:
        inference_speed = frame_count / total_inference_time
        print(f'Total model inference time: {total_inference_time:.2f} seconds')
        print(f'Model inference speed: {inference_speed:.2f} frames/sec')

    print(f'Total script runtime: {end_time - start_time:.2f} seconds')
    print('Prediction complete.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict trajectories on a video using a trained model")
    parser.add_argument('--video_path', required=True, help="Path to the video file")
    parser.add_argument('--model_weights', required=True, help="Path to the model weights")
    parser.add_argument('--output_dir', default=os.path.join(os.getcwd(), 'predicts_demo'), help="Directory to save output files (default: current working directory)")
    parser.add_argument('--queue_length', type=int, default=8, help="Length of the predicted points queue (default: 8)")
    parser.add_argument('--show_frame', action='store_true', help="Display the processed frames during inference (default: False)")
    args = parser.parse_args()
    main(args)
