# IntelliCount: AI Object Detection for Vehicle Tracking and Traffic Insights

A computer vision system that detects, counts, and tracks vehicles in traffic videos using YOLOv8, developed as a final year B.Tech capstone project.

## Project Overview

This project uses state-of-the-art object detection to analyze traffic footage. It can identify cars, trucks, buses, and motorcycles in videos, and provide real-time statistics and comprehensive reports. IntelliCount was developed to create an efficient solution for traffic monitoring and analysis using AI-powered detection techniques.

## Features

- **Vehicle Detection & Classification**: Identifies cars, trucks, buses, and motorcycles
- **Real-Time Statistics**: Shows current frame statistics in an overlay
- **Vehicle Counting**: Keeps track of all vehicles seen in the video
- **Visual Dashboard**: Displays live counts and totals in a dashboard
- **Report Generation**: Creates PDF and JSON reports after processing
- **Command-Line Interface**: Supports various options to customize the analysis

## Installation

### Prerequisites

- Python 3.9 or higher
- Conda (recommended for environment management)

### Setup Options

#### Option 1: Using the environment.yml file (Recommended)

This is the fastest way to set up the complete environment:

```bash
# Clone the repository
git clone https://github.com/azadraval/IntelliCount-AI-Object-Detection-for-Vehicle-Tracking-and-Traffic-Insights.git
cd IntelliCount-AI-Object-Detection-for-Vehicle-Tracking-and-Traffic-Insights

# Create and activate the conda environment from the yml file
conda env create -f environment.yml
conda activate env
```

#### Option 2: Using requirements.txt

If you prefer using pip or a virtual environment:

```bash
# Clone the repository
git clone https://github.com/azadraval/IntelliCount-AI-Object-Detection-for-Vehicle-Tracking-and-Traffic-Insights.git
cd IntelliCount-AI-Object-Detection-for-Vehicle-Tracking-and-Traffic-Insights

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 3: Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/azadraval/IntelliCount-AI-Object-Detection-for-Vehicle-Tracking-and-Traffic-Insights.git
   cd IntelliCount-AI-Object-Detection-for-Vehicle-Tracking-and-Traffic-Insights
   ```

2. Create and activate a conda environment:
   ```
   conda create -n env python=3.9
   conda activate env
   ```

3. Install the required packages:
   ```
   pip install opencv-python ultralytics numpy matplotlib
   ```

#### Important: Set OpenMP Environment Variable

Regardless of which installation method you use, you must set the OpenMP environment variable to avoid runtime conflicts:

```
# In PowerShell:
[System.Environment]::SetEnvironmentVariable('KMP_DUPLICATE_LIB_OK', 'TRUE', 'User')

# Or in Command Prompt:
setx KMP_DUPLICATE_LIB_OK TRUE
```

### Included Dependencies

The project relies on the following key packages:

- **Core Requirements**:
  - `ultralytics`: YOLOv8 implementation for object detection
  - `opencv-python`: Computer vision operations and video processing
  - `numpy`: Numerical computing and array operations
  - `torch` & `torchvision`: Deep learning framework

- **Visualization & Analysis**:
  - `matplotlib`: Creating reports and visualizations
  - `seaborn`: Statistical data visualization
  - `pandas`: Data manipulation and analysis

- **Utilities**:
  - `tqdm`: Progress bars for processing feedback
  - `PyYAML`: Configuration file handling

For a complete list of dependencies, see the `requirements.txt` file.

## Usage

### Using the Batch File

The easiest way to run the system is to use the included batch file:

```
.\run_traffic_analysis.bat
```

### Manual Execution

To run the system manually with default settings:

```
conda activate env
python main.py
```

### Command-Line Options

The system supports various command-line options:

```
python main.py --video my_video.mp4 --model yolov8l.pt --conf 0.5 --device cuda
```

#### Available Options

- `--video`: Path to input video (default: "8698-213454544.mp4")
- `--model`: Path to YOLOv8 model (default: "yolov8m.pt")
- `--conf`: Confidence threshold (default: 0.45)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--device`: Device to run on - "cpu" or "cuda" (default: "cpu")
- `--no-display`: Disable display window
- `--report`: Generate reports (default: enabled)

## Output

The system generates:

1. An annotated video with detection boxes
2. A JSON report with detection statistics
3. A PDF report with detection statistics
4. Console output with real-time detection information

All outputs are saved to the `outputs` directory.

## For Developers

If you're interested in using this project as a foundation for your own work or contributing to its development, this section provides additional guidance.

### Detailed Installation Guide

1. **Full Dependencies**: For a complete development environment, install the following:
   ```
   pip install opencv-python ultralytics numpy matplotlib torch torchvision tqdm pillow seaborn pandas
   ```

2. **Get Sample Videos**: If you don't have traffic footage to test with:
   - Sample traffic videos can be downloaded from [Pexels](https://www.pexels.com/search/videos/traffic/)
   - Place video files in the project root directory or specify their path using the `--video` parameter

3. **Alternative YOLOv8 Models**: For different performance/accuracy tradeoffs:
   ```
   # Faster but less accurate
   pip install yolov8n.pt
   
   # More accurate but slower
   pip install yolov8l.pt
   pip install yolov8x.pt
   ```

### Project Structure Explanation

- `main.py`: The core file containing all system functionality:
  - Vehicle detection and tracking algorithms
  - Dashboard and visualization components
  - Report generation functionality
  - Command-line interface

- `run_traffic_analysis.bat`: Convenience script to run the system

- `outputs/`: Directory for all generated files

### Modifying the Project

#### Adding New Vehicle Types

To detect additional vehicle types:
1. Open `main.py`
2. Find the model inference section (around line 375)
3. Adjust the classes parameter in `results = model(frame, classes=[2, 3, 5, 7], conf=confidence_threshold, ...)`
4. Add the new vehicle type to the color mapping and statistics dictionaries

#### Customizing the Dashboard

To modify the dashboard appearance:
1. Locate the `create_dashboard()` function (around line 87)
2. Adjust colors, layout, or add new statistics as needed

#### Adding a New Feature

Example: Adding a speed estimation feature:
1. Create a new function in `main.py`:
   ```python
   def estimate_speed(previous_position, current_position, fps):
       """Estimates speed of a vehicle based on position change"""
       # Calculate distance moved in pixels
       distance = np.sqrt((current_position[0] - previous_position[0])**2 + 
                          (current_position[1] - previous_position[1])**2)
       
       # Convert pixels to a real-world measure (requires calibration)
       # This is a placeholder - real implementation would need camera calibration
       pixel_to_meter = 0.1  # Example: 10 pixels = 1 meter
       
       # Calculate speed: distance/time (where time = 1/fps)
       speed_ms = distance * pixel_to_meter * fps
       speed_kmh = speed_ms * 3.6
       
       return speed_kmh
   ```

2. Update the tracking system to use this function
3. Modify the dashboard to display speed information

### Performance Optimization

To improve performance:
1. Use a smaller YOLOv8 model (like yolov8n.pt)
2. Reduce input frame resolution: add a resize step before detection
3. Process only every Nth frame for higher FPS
4. Enable GPU acceleration with `--device cuda` (requires CUDA support)

### Building a Standalone Application

To create a standalone executable:
1. Install PyInstaller:
   ```
   pip install pyinstaller
   ```

2. Create the executable:
   ```
   pyinstaller --onefile --add-data "yolov8m.pt;." main.py
   ```

3. The executable will be in the `dist` directory

## Troubleshooting

### OpenMP Errors

If you encounter errors related to OpenMP ("libiomp5md.dll already initialized"), set the environment variable:

```
set KMP_DUPLICATE_LIB_OK=TRUE
```

### CUDA/GPU Issues

For GPU-related problems, make sure you have the correct CUDA version installed for your PyTorch/Ultralytics installation.

### Common Issues and Solutions

1. **Video Loading Fails**: Check video file format and codec compatibility with OpenCV
2. **Model Not Found**: Ensure YOLOv8 model file is in the correct location or specify full path with `--model`
3. **Low Performance**: Try reducing video resolution or using a more efficient model
4. **Memory Errors**: Reduce batch size or video resolution if running out of memory

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection model
- [OpenCV](https://opencv.org/) for image processing capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Feel free to contact me for questions or collaborations on this project.

---
Created as a Final Year B.Tech Capstone Project