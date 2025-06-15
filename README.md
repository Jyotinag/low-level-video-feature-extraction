# Video Feature Extraction for VR Cybersickness Prediction

This repository contains code and methodology for extracting low-level video features from virtual reality (VR) gameplay data. These features are intended for use in cybersickness prediction models using datasets like **VRWalking** and **SET**.

---

## Overview

We extract 11 video-based features from VR gameplay recordings to understand their correlation with cybersickness (measured by FMS scores). The videos were collected using OBS Studio, and the left-eye video stream was recorded. 

- For the **VRWalking dataset**, videos were originally recorded at 60 FPS. We downsampled these to 25 FPS by randomly selecting 25 frames from each 60-frame chunk, to match the frame rate of the **SET dataset**, which was already at 25 FPS.
- Features are extracted per frame or per frame pair and reduced to scalar values (e.g., by averaging across pixels or directions), making them interpretable and suitable for machine learning models.

---

## Scripts

### 1. `process_video.ipynb` & `parallel_process.ipynb`

These Jupyter notebooks extract video features directly from `.mkv` or `.mp4` video files. They handle tasks like:

- Reading video frames
- Computing features (optical flow, edge intensity, etc.)
- Saving feature vectors in structured CSV format

### 2. `video_data_preprocess.py`

This script extracts features from **timestamped image frames** rather than full video files. It is used when frames are already extracted and labeled with precise timestamps.

---

## Extracted Video Features

### 1. Optical Flow
Measures the amount and direction of motion between consecutive frames. We calculate average motion magnitude over all pixels. High motion is often associated with increased cybersickness.

### 2. Histogram of Oriented Gradients (HOG)
Captures object shape and appearance by measuring edge orientations in small image patches. HOG can reveal visual clutter or background complexity, which may affect cybersickness.

### 3. Color Histogram
Represents the distribution of pixel colors. This captures the color composition of the video, which can influence user comfort. For example, highly saturated or rapidly changing colors might induce discomfort.

### 4. Edge Intensity
Measures sharpness and the strength of object boundaries in a frame. Sudden or strong edges may correlate with scene complexity and potentially increase cybersickness.

### 5. Scene Cuts
Detects transitions between scenes based on differences between consecutive frames. These abrupt changes can interrupt visual flow and have an impact on comfort.

### 6. Temporal Smoothness
Quantifies how smoothly pixel values transition across frames. Lower temporal smoothness suggests jarring or abrupt changes, which can increase motion discomfort.

### 7. Brightness Flicker
Measures fluctuations in average brightness over time. Flickering brightness can strain the eyes and contribute to visual discomfort or cybersickness.

### 8. Spectral Entropy
Captures randomness or complexity in the frequency domain of a video frame. Higher entropy suggests more chaotic content, which may affect perceptual stability.

### 9. Spatial Frequency
Represents how often patterns or textures repeat across an image. High spatial frequency indicates high detail, which may either improve clarity or contribute to visual overload.

### 10. Luminance and Contrast
Luminance reflects overall brightness; contrast measures the difference between light and dark areas. Lower luminance and balanced contrast levels are generally more comfortable for viewers.

### 11. Time Series
Each frame is associated with a timestamp, allowing temporal alignment of video features with cybersickness scores. This also helps in modeling how sickness evolves over time.

---

## Citation

If you use this code or feature extraction strategy, please cite the original sources and related works such as:

- Sanaei et al., 2024 - Correlations between motion cues and cybersickness
- Oh et al., 2022 - Background complexity and cybersickness
- So et al., 2007 - Effects of color and visual variables on sickness
- Rahimi et al., 2018 - Scene features and visual fatigue
- Chang et al., 2013 - Scene breaks and motion sickness
- Palmisano et al., 2017 - Smooth motion and vection
- Vasylevska et al., 2019 - Lighting and VR comfort

---
