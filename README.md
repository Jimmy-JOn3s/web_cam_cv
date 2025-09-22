# Image Processing Application

A real-time image processing application with webcam integration, built using OpenCV and organized with Object-Oriented Programming principles.

## Features

- **Real-time Webcam Processing**: Live webcam feed with real-time image processing
- **Color Space Conversion**: Switch between BGR, Grayscale, and HSV color spaces
- **Brightness & Contrast Adjustment**: Real-time brightness and contrast control
- **Image Filtering**: 
  - Gaussian Blur with adjustable parameters
  - Bilateral Filter with adjustable parameters
- **Histogram Analysis**: Live histogram display and analysis
- **Modular Design**: Easy to extend with new filters and processing functions

## Project Structure

```
task1/
├── main.py                          # Main application entry point
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── test.py                         # Original test file (legacy)
└── src/                           # Source code modules
    ├── core/                      # Core functionality
    │   ├── webcam_manager.py      # Webcam operations
    │   └── image_processor.py     # Main image processing logic
    ├── filters/                   # Image filters
    │   ├── base_filter.py         # Abstract base class for filters
    │   ├── gaussian_filter.py     # Gaussian blur implementation
    │   └── bilateral_filter.py    # Bilateral filter implementation
    ├── color_spaces/              # Color space operations
    │   └── histogram_analyzer.py  # Histogram analysis and visualization
    └── ui/                        # User interface components
        ├── trackbar_manager.py    # Trackbar management
        └── display_manager.py     # Window and display management
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

### Controls

- **Trackbars**: Use the trackbars in the main window to adjust parameters in real-time
- **Keyboard Shortcuts**:
  - `1`: Switch to BGR color space
  - `2`: Switch to Grayscale
  - `3`: Switch to HSV color space
  - `h`: Toggle histogram display
  - `g`: Toggle Gaussian filter
  - `b`: Toggle Bilateral filter
  - `r`: Reset all parameters to defaults
  - `s`: Save current frame
  - `q`: Quit application

### Trackbar Controls

- **Brightness**: Adjust image brightness (-100 to +100)
- **Contrast**: Adjust image contrast (0.01 to 2.0)
- **Color Space**: Select color space (0=BGR, 1=Grayscale, 2=HSV)
- **Gaussian Kernel**: Kernel size for Gaussian blur
- **Gaussian Sigma X/Y**: Standard deviation for Gaussian blur
- **Bilateral d**: Diameter for bilateral filter
- **Bilateral Sigma Color**: Color space sigma for bilateral filter
- **Bilateral Sigma Space**: Coordinate space sigma for bilateral filter
- **Enable Gaussian**: Toggle Gaussian filter on/off
- **Enable Bilateral**: Toggle Bilateral filter on/off
- **Show Histogram**: Toggle histogram display on/off

## Architecture

### Core Components

1. **WebcamManager**: Handles webcam initialization, frame capture, and resource management
2. **ImageProcessor**: Main processing engine with brightness/contrast adjustment and color space conversion
3. **HistogramAnalyzer**: Calculates and visualizes image histograms

### Filter System

- **BaseFilter**: Abstract base class providing a common interface for all filters
- **GaussianFilter**: Implements Gaussian blur with configurable parameters
- **BilateralFilter**: Implements edge-preserving bilateral filtering

### UI System

- **TrackbarManager**: Manages OpenCV trackbars for real-time parameter adjustment
- **DisplayManager**: Handles window creation, image display, and text overlays

## Extending the Application

### Adding New Filters

1. Create a new filter class inheriting from `BaseFilter`:
```python
from src.filters.base_filter import BaseFilter

class MyNewFilter(BaseFilter):
    def __init__(self):
        super().__init__("My New Filter")
        self.parameters = {'param1': default_value}
    
    def apply(self, image, **kwargs):
        # Implement your filter logic
        return processed_image
    
    def get_parameter_info(self):
        # Return parameter information
        return parameter_info_dict
```

2. Add the filter to the main application in `main.py`
3. Create trackbars for the new filter parameters

### Adding New Color Spaces

Add new color space conversions in `ImageProcessor.convert_color_space()` method.

### Adding New Analysis Tools

Create new analysis classes similar to `HistogramAnalyzer` in the `color_spaces` directory.

## Dependencies

- **OpenCV (cv2)**: Computer vision and image processing
- **NumPy**: Numerical computations
- **Matplotlib**: Plotting and visualization (for advanced histogram features)

## Notes

- The application automatically detects and handles different webcam resolutions
- All filters preserve the original image data type and format
- The modular design allows easy testing of individual components
- Error handling is implemented throughout the application
