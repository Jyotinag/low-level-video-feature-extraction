import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.filters import sobel
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import random
import os
import re

def capture_frames_from_folder(folder_path, target_fps=10, crop_ratio=0.8):
    # List all PNG files in the folder
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # Sort the files based on the numeric part of the filename (e.g., Frame-977-...)
    png_files.sort(key=lambda x: int(re.search(r'Frame-(\d+)', x).group(1)))

    frames = []
    frame_count = 0

    # Calculate the probability of selecting a frame to meet the target fps
    select_probability = target_fps / 25  # Assuming original FPS is 60 for the frames

    for png_file in png_files:  # Files are now sorted by frame number
        frame = cv2.imread(os.path.join(folder_path, png_file))

        # Randomly select frames to match target fps
        if random.random() < select_probability:
            # Get frame dimensions
            h, w, _ = frame.shape

            # Compute the crop dimensions (80% of the center)
            crop_h = int(h * crop_ratio)
            crop_w = int(w * crop_ratio)
            start_y = (h - crop_h) // 2
            start_x = (w - crop_w) // 2

            # Crop the frame to the central 80%
            cropped_frame = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]

            frames.append(cropped_frame)

        frame_count += 1

    return frames

# The feature extraction functions remain unchanged; you can use the same ones as before.

def extract_optical_flow(frames, fps):
    print("Start Optical Flow")
    prev_frame = None
    optical_flows = []

    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            optical_flows.append(np.mean(flow_magnitude))
        prev_frame = gray_frame

    return downsample_to_1hz(optical_flows, fps)

def extract_hog_features(frames, fps):
    print("Start HOG")
    hog_features = []

    for frame in frames:
        gray_frame = rgb2gray(frame)
        hog_feature = hog(gray_frame, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
        hog_features.append(np.mean(hog_feature))

    return downsample_to_1hz(hog_features, fps)

def extract_color_histogram(frames, fps):
    print("Start Color Histogram")
    color_histograms = []

    for frame in frames:
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
        color_hist = np.concatenate((hist_b, hist_g, hist_r)).flatten()
        color_histograms.append(np.mean(color_hist))

    return downsample_to_1hz(color_histograms, fps)

def extract_edge_detection(frames, fps):
    print("Start Edge Detection")
    edge_intensities = []

    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = sobel(gray_frame)
        edge_intensity = np.mean(edges)
        edge_intensities.append(edge_intensity)

    return downsample_to_1hz(edge_intensities, fps)

def extract_scene_cuts(frames, fps):
    print("Start Scene Cuts")
    prev_hist = None
    scene_cuts = []
    cuts_per_second = 0

    for i, frame in enumerate(frames):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = np.sum(np.abs(hist - prev_hist))
            if diff > 0.4:  # Threshold to detect scene cuts
                cuts_per_second += 1

        prev_hist = hist

        if (i + 1) % fps == 0:
            scene_cuts.append(cuts_per_second)
            cuts_per_second = 0

    return scene_cuts

def extract_temporal_smoothness(frames, fps):
    print("Start Temporal Smoothness")
    prev_frame = None
    smoothness_values = []

    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = np.mean(np.abs(gray_frame - prev_frame))
            smoothness_values.append(diff)
        prev_frame = gray_frame

    return downsample_to_1hz(smoothness_values, fps)

def extract_flicker_brightness(frames, fps):
    print("Start Flicker Brightness")
    brightness_flickers = []
    prev_brightness = None

    for frame in frames:
        brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if prev_brightness is not None:
            flicker = abs(brightness - prev_brightness)
            brightness_flickers.append(flicker)
        prev_brightness = brightness

    return downsample_to_1hz(brightness_flickers, fps)

def extract_spectral_entropy(frames, fps):
    print("Start Spectral Entropy")
    spectral_entropies = []

    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fourier_transform = np.fft.fft2(gray_frame)
        magnitude_spectrum = np.abs(fourier_transform)
        spectral_entropy = shannon_entropy(magnitude_spectrum)
        spectral_entropies.append(spectral_entropy)

    return downsample_to_1hz(spectral_entropies, fps)

def extract_spatial_frequency(frames, fps):
    print("Start Spatial Frequency")
    spatial_frequencies = []

    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frequency_spectrum = np.fft.fftshift(np.fft.fft2(gray_frame))
        magnitude_spectrum = np.abs(frequency_spectrum)
        spatial_frequency = np.mean(magnitude_spectrum)
        spatial_frequencies.append(spatial_frequency)

    return downsample_to_1hz(spatial_frequencies, fps)

def extract_luminance_contrast(frames, fps):
    print("Start Luminance and Contrast")
    luminance_values = []
    contrast_values = []

    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luminance = np.mean(gray_frame)
        contrast = gray_frame.std()
        luminance_values.append(luminance)
        contrast_values.append(contrast)

    luminance_downsampled = downsample_to_1hz(luminance_values, fps)
    contrast_downsampled = downsample_to_1hz(contrast_values, fps)

    return luminance_downsampled, contrast_downsampled

def extract_texture_features(frames, fps):
    print("Start Texture")
    texture_contrasts = []

    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray_frame, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        texture_contrast = graycoprops(glcm, 'contrast')[0, 0]
        texture_contrasts.append(texture_contrast)

    return downsample_to_1hz(texture_contrasts, fps)

def downsample_to_1hz(feature_values, fps):
    """Downsamples the feature values to 1Hz (1 value per second)."""
    return [np.mean(feature_values[i:i + fps]) for i in range(0, len(feature_values), fps)]

def process_frame_features(folder_path):
    frames = capture_frames_from_folder(folder_path)

    fps = 10  # Assume a target FPS (if required, you can set it to match original FPS)

    # Store each feature in variables first
    optical_flow_feature = extract_optical_flow(frames, fps)
    hog_feature = extract_hog_features(frames, fps)
    color_histogram_feature = extract_color_histogram(frames, fps)
    edge_intensity_feature = extract_edge_detection(frames, fps)
    scene_cuts_feature = extract_scene_cuts(frames, fps)
    temporal_smoothness_feature = extract_temporal_smoothness(frames, fps)
    brightness_flicker_feature = extract_flicker_brightness(frames, fps)
    spectral_entropy_feature = extract_spectral_entropy(frames, fps)
    spatial_frequency_feature = extract_spatial_frequency(frames, fps)
    luminance_feature, contrast_feature = extract_luminance_contrast(frames, fps)

    # Create a dictionary to hold all features
    max_length = max(
        len(optical_flow_feature), len(hog_feature), len(color_histogram_feature),
        len(edge_intensity_feature), len(scene_cuts_feature), len(temporal_smoothness_feature),
        len(brightness_flicker_feature), len(spectral_entropy_feature), len(spatial_frequency_feature),
        len(luminance_feature), len(contrast_feature)
    )

    # Pad all features with NaN to match the maximum length
    def pad_feature(feature, max_length):
        feature = np.array(feature, dtype=float)
        return np.pad(feature, (0, max_length - len(feature)), mode='constant', constant_values=np.nan)

    features = {
        "optical_flow": pad_feature(optical_flow_feature, max_length),
        "hog_features": pad_feature(hog_feature, max_length),
        "edge_intensity": pad_feature(edge_intensity_feature, max_length),
        "scene_cuts": pad_feature(scene_cuts_feature, max_length),
        "temporal_smoothness": pad_feature(temporal_smoothness_feature, max_length),
        "brightness_flicker": pad_feature(brightness_flicker_feature, max_length),
        "spectral_entropy": pad_feature(spectral_entropy_feature, max_length),
        "spatial_frequency": pad_feature(spatial_frequency_feature, max_length),
        "luminance": pad_feature(luminance_feature, max_length),
        "contrast": pad_feature(contrast_feature, max_length)
    }

    # Create a DataFrame from the feature dictionary
    feature_df = pd.DataFrame(features)

    return feature_df

# Example usage
folder_path = r'E:\Projects\Projects\User Study Data\Raw Data\walk\08_132569490239789285\WalkingBeach\Frames'
feature_df = process_frame_features(folder_path)
feature_df.to_csv("8_video.csv")
print("################File Processed###################")