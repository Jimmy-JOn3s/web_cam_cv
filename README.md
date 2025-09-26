# Computer Vision Application

Real-time computer vision application with webcam integration, featuring image processing, panorama creation, augmented reality, and camera calibration.

## Setup

### 1. Install Miniforge and Create Conda Environment

This application uses Miniforge and Conda environments for dependency management.

**Step 1: Install Miniforge**
```bash
# macOS with Homebrew:
brew install --cask miniforge

# Or download from: https://github.com/conda-forge/miniforge
```

**Step 2: Create and Activate Environment**
```bash
conda create -n cv_app python=3.9
conda activate cv_app
```

**Step 3: Install Required Packages**
```bash
conda install -c conda-forge opencv numpy matplotlib
```

**Step 4: Run the Application**
```bash
python main.py
```

### Requirements
- Python 3.9+
- OpenCV
- NumPy 
- Matplotlib
- Webcam (USB or built-in)

## User Manual

### Keyboard Controls

#### Basic Controls
- `q` - Quit application
- `r` - Reset all settings to default
- `s` - Save current frame as image

#### Color Spaces
- `1` - BGR color mode (default)
- `2` - Grayscale mode
- `3` - HSV color mode

#### Image Processing Filters
- `h` - Toggle histogram display
- `g` - Toggle Gaussian blur filter
- `b` - Toggle bilateral filter
- `e` - Toggle Canny edge detection
- `l` - Toggle Hough line detection
- `t` - Toggle geometric transformations

#### Camera Calibration
- `c` - Enter/exit calibration mode
- `SPACE` - Capture calibration image (in calibration mode)

#### Panorama Creation
- `p` - Enter/exit panorama mode
- `SPACE` - Capture frame for panorama (in panorama mode)
- `m` - Create panorama from captured frames
- `ESC` - Exit panorama mode

#### Augmented Reality
- `a` - Enter/exit AR mode
- `v` - Change render mode (wireframe/solid/both)
- `n` - Toggle T-Rex animation
- `x` - Toggle coordinate axis display

### How to Use

#### 1. Basic Image Processing
1. Run `python main.py`
2. Use number keys `1`, `2`, `3` to change color spaces
3. Adjust brightness/contrast with trackbars
4. Press filter keys `g`, `b`, `e`, `l`, `t` to enable filters

#### 2. Panorama Creation
1. Press `p` to enter panorama mode
2. Press `SPACE` to capture first frame
3. Move camera horizontally with 30% overlap
4. Press `SPACE` to capture more frames
5. Press `m` to create final panorama
6. Panoramas save to `panorama/` folder

#### 3. Camera Calibration  
1. Press `c` to enter calibration mode
2. Point camera at chessboard pattern
3. Application auto-captures when pattern detected
4. Move to different angles for variety
5. Calibration completes automatically

#### 4. Augmented Reality
1. Press `a` to enter AR mode
2. Point camera at ArUco marker
3. T-Rex model appears on marker
4. Use `v`, `n`, `x` to control display options

### Output Files
- **Panoramas**: `panorama/panorama_YYYYMMDD_HHMMSS.jpg` (full resolution)
- **Thumbnails**: `panorama/thumb_panorama_YYYYMMDD_HHMMSS.jpg` (quick preview)
- **Saved Frames**: `processed_frame_[timestamp].jpg` (press `s` to save)
