# Squash Player Pose Detection

This project uses the YOLOv8-pose model to detect the positions of squash players in a YouTube video. It generates an animated video with the players' key limb positions highlighted and a CSV file with the keypoint data.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- `pip` (Python package installer)
- `ffmpeg`: This is a crucial system dependency required for processing video and audio.

### Installing `ffmpeg`

- **On Debian/Ubuntu:**
  ```bash
  sudo apt-get update
  sudo apt-get install ffmpeg
  ```

- **On macOS (using Homebrew):**
  ```bash
  brew install ffmpeg
  ```

- **On Windows:**
  Download the latest build from the [official FFmpeg website](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH.

## Setup and Execution

Follow these steps to run the project:

**1. Clone the Repository**

First, clone this repository to your local machine.

**2. Install Python Dependencies**

Navigate to the project directory and install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**3. Run the Script**

Execute the main script. It will automatically download the YouTube video (if it's not already present) and then process it to generate the output files.

```bash
python process_video.py
```

## Output

After the script finishes, you will find two new files in your project directory:

- `animation.mp4`: A video file showing the original squash game with the players' skeletons (limbs and keypoints) drawn on top.
- `keypoints.csv`: A CSV file containing the raw coordinate data for each detected keypoint for every person in every frame.
