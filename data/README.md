# Data Directory

This directory contains the image data for the HCMAI 2025 Image Search project.

## Structure

```
data/
├── keyframes/          # Extracted video keyframes
│   ├── video1/         # Frames from video1
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   │   └── ...
│   └── video2/
│       └── ...
└── videos/             # Original video files (optional)
    ├── video1.mp4
    └── video2.mp4
```

## Usage

1. Place your video files in the `videos` directory (optional)
2. Run the keyframe extraction script:
   ```bash
   python extract_keyframes.py --input_dir data/videos --output_dir data/keyframes
   ```
3. The application will automatically serve images from the `keyframes` directory

## Notes

- The `keyframes` directory should contain subdirectories for each video
- Each subdirectory should contain the extracted frames as JPG or PNG files
- The application will ignore this README file and `.gitkeep` file
