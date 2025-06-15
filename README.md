# Video Feature Extraction for VR Gameplay Analysis

This repository contains tools and scripts for extracting low-level video features from VR gameplay data to support cybersickness prediction using objective visual features.

## Overview

The project focuses on extracting 11 interpretable video features from VR video data, downsampled to 25 FPS to maintain consistency across datasets (VRWalking and SET). These features are selected based on their relevance to motion, brightness, color, and structural characteristics—all of which are known to influence cybersickness.

### Key Features Extracted

1. **Optical Flow**  
   Measures motion by calculating the pixel displacement between consecutive frames:
   $$
   V(x, y) = (\Delta x, \Delta y)
   $$
   $$
   \text{Average Optical Flow} = \frac{1}{N} \sum_{i=1}^{N} \sqrt{(\Delta x_i)^2 + (\Delta y_i)^2}
   $$

2. **Histogram of Oriented Gradients (HOG)**  
   Captures the gradient orientation and magnitude:
   $$
   G_x = I(x+1, y) - I(x-1, y), \quad G_y = I(x, y+1) - I(x, y-1)
   $$
   $$
   |\mathbf{G}(x, y)| = \sqrt{G_x^2 + G_y^2}, \quad \theta(x, y) = \arctan\left(\frac{G_y}{G_x}\right)
   $$

3. **Color Histogram**  
   Calculates pixel frequency for each color bin:
   $$
   H_c(b) = \sum_{(x, y)} \delta(I_c(x, y) - b)
   $$

4. **Edge Intensity**  
   Uses the Sobel operator:
   $$
   G_x = \sum I(x+i, y+j) \cdot S_x(i, j), \quad G_y = \sum I(x+i, y+j) \cdot S_y(i, j)
   $$
   $$
   |\mathbf{G}| = \sqrt{G_x^2 + G_y^2}
   $$

5. **Scene Cuts**  
   Measures frame-wise changes to detect transitions:
   $$
   D(t) = \frac{1}{N} \sum_{(x, y)} \left( I_t(x, y) - I_{t+1}(x, y) \right)^2
   $$
   $$
   \text{Scene Cut} = \begin{cases} 
   1 & \text{if } D(t) > \tau \\
   0 & \text{otherwise}
   \end{cases}
   $$

6. **Temporal Smoothness**  
   Measures pixel intensity consistency:
   $$
   S(t) = \frac{1}{N} \sum_{(x, y)} \left| I_t(x, y) - I_{t+1}(x, y) \right|
   $$

7. **Brightness Flicker**  
   Measures temporal brightness variance:
   $$
   B(t) = \frac{1}{N} \sum_{(x, y)} I_t(x, y)
   $$
   $$
   F(t) = |B(t) - B(t-1)|
   $$

8. **Spectral Entropy**  
   Computed from the frequency domain:
   $$
   E = - \sum_{f} P(f) \log P(f)
   $$

9. **Spatial Frequency**  
   Captures texture and detail:
   $$
   SF = \sqrt{\left( \frac{1}{N} \sum G_x^2 \right) + \left( \frac{1}{N} \sum G_y^2 \right)}
   $$

10. **Luminance and Contrast**  
   Measures brightness and intensity contrast:
   $$
   L = \frac{1}{N} \sum_{(x, y)} I(x, y)
   $$
   $$
   C = \frac{I_{\text{max}} - I_{\text{min}}}{I_{\text{max}} + I_{\text{min}}}
   $$

11. **Time Series**  
   Timestamps are used to track temporal progression and session duration for participants.

---

## File Descriptions

### `process_video.ipynb`  
Performs feature extraction on video files (`.mkv` or `.mp4`). This notebook processes video data frame-by-frame to extract motion, color, brightness, and structural cues.


### `video_data_preprocess.py`  
Processes already timestamped frames (e.g., extracted left-eye views or simulation recordings) and extracts features directly from image folders with associated metadata.

---

## Usage

### Requirements

- Python 3.8+
- OpenCV
- NumPy
- SciPy
- scikit-image
- tqdm
- Jupyter (for notebooks)

Install dependencies:
```bash
pip install -r requirements.txt
