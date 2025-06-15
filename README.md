# Video Feature Extraction for Cybersickness Analysis in VR

This repository provides the methodology and implementation details for extracting low-level video features from VR gameplay data, specifically tailored to study and predict cybersickness. The extracted features are designed to be interpretable and correlated with motion-induced discomfort in virtual environments.

## Dataset Overview

We used two datasets:
- **VRWalking Dataset**: VR gameplay was recorded at **60 FPS** using OBS Studio, focusing on left-eye video frames.
- **SET Dataset**: Gameplay recorded at **25 FPS**.

To ensure consistency, we **downsampled VRWalking videos to 25 FPS** by randomly selecting 25 frames from each 60-frame window.

---

## Extracted Features

The following **11 features** were extracted from the VR videos. For multidimensional features (e.g., optical flow), we computed the **average across dimensions** to create a single representative value per frame or frame pair.

### 1. Optical Flow
- **Purpose**: Measures motion intensity between consecutive frames.
- **Computation**: Calculates displacement vectors \( (\Delta x, \Delta y) \) for each pixel and summarizes via average magnitude.
- **Equation**:
  \[
  \text{Average Optical Flow} = \frac{1}{N} \sum_{i=1}^{N} \sqrt{(\Delta x_i)^2 + (\Delta y_i)^2}
  \]

### 2. Histogram of Oriented Gradients (HOG)
- **Purpose**: Captures object shapes and contours via gradient orientation histograms.
- **Computation**: Gradients are calculated and binned into histograms.
- **Equation**:
  \[
  \theta(x, y) = \arctan\left(\frac{G_y}{G_x}\right)
  \]

### 3. Color Histogram
- **Purpose**: Represents distribution of colors in the frame.
- **Computation**: Counts pixel intensities per bin per color channel (R, G, B).

### 4. Edge Intensity
- **Purpose**: Measures strength of edges using Sobel filters.
- **Computation**: Applies horizontal and vertical Sobel kernels and computes the gradient magnitude.

### 5. Scene Cuts
- **Purpose**: Detects transitions between different visual scenes.
- **Computation**: Calculates mean squared difference between frames and applies a threshold.
- **Threshold**: \( \tau = 0.4 \)

### 6. Temporal Smoothness
- **Purpose**: Measures consistency of motion or transitions.
- **Computation**: Calculates frame-to-frame intensity difference.

### 7. Brightness Flicker
- **Purpose**: Measures fluctuation in brightness, which may cause discomfort.
- **Computation**: Computes difference in average intensity over consecutive frames.

### 8. Spectral Entropy
- **Purpose**: Quantifies randomness in frequency domain.
- **Computation**: Uses normalized power spectrum from Fourier Transform.
- **Equation**:
  \[
  E = - \sum_{f} P(f) \log P(f)
  \]

### 9. Spatial Frequency
- **Purpose**: Measures how often patterns repeat (level of detail).
- **Computation**: Derived from gradient energies in the spatial domain.

### 10. Luminance and Contrast
- **Luminance**: Average brightness from grayscale image.
- **Contrast**: Normalized intensity range:
  \[
  C = \frac{I_{\text{max}} - I_{\text{min}}}{I_{\text{max}} + I_{\text{min}}}
  \]

### 11. Time Series (Timestamps)
- **Purpose**: Tracks time spent in simulation per participant.
- **Use Case**: Useful for analyzing how duration affects cybersickness (FMS score).

---

## References

This work builds on prior research on cybersickness-related feature extraction:

- Sanaei et al. (2024) – Optical Flow and Cybersickness Correlation.
- Oh et al. (2022) – HOG Features and Visual Comfort.
- So et al. (2007) – Impact of Color on Cybersickness.
- Rahimi et al. (2018) – Scene Structure and Motion Effects.
- Chang et al. (2013) – Scene Cuts and Rest Frames.
- Palmisano et al. (2017) – Motion Smoothness and Vection.
- Vasylevska et al. (2019) – Luminance, Contrast, and Discomfort.

---

## Usage

This repository provides Python scripts to extract and process the above features using standard video processing libraries (e.g., OpenCV, scikit-image). Each feature can be extracted independently and saved in a CSV for further analysis with cybersickness labels (e.g., FMS scores).

Please see `process.py` (to be added) for implementation details.

---

## License

This work is intended for academic research only. Please cite appropriately if used in publications.
