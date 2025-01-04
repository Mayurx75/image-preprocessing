# Image Processing with Python

This project demonstrates various image processing techniques using Python libraries like OpenCV, Matplotlib, PIL, and imgaug. The code includes operations such as basic image transformations, histogram analysis, image augmentation, and advanced image enhancements.

## Table of Contents
- [Requirements](#requirements)
- [Features](#features)
- [Code Overview](#code-overview)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Requirements
To run the code, you need the following libraries:

- Python 3.6 or later
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- Pillow (`PIL`)
- imgaug
- imageio

Install the dependencies using pip:
```bash
pip install opencv-python numpy matplotlib pillow imgaug imageio
```

## Features
- **Basic Image Operations**:
  - Loading images in grayscale and color.
  - Rotating, flipping, resizing, and cropping images.
  - Adding Gaussian noise and changing brightness.

- **Histogram Analysis**:
  - Calculating and plotting grayscale histograms.
  - Histogram equalization and CLAHE (Contrast Limited Adaptive Histogram Equalization).

- **Augmentation**:
  - Horizontal flips, random cropping, contrast adjustment, rotation, and noise addition.

- **Advanced Transformations**:
  - Shearing transformations.
  - Contrast enhancement in different color spaces (e.g., YUV).

## Code Overview
The code is divided into sections:

1. **Basic Image Analysis**:
   - Load images and find pixel intensity values.

2. **Transformations**:
   - Rotate, flip, crop, and resize images.

3. **Augmentation**:
   - Use the `imgaug` library for augmentations.

4. **Histogram Operations**:
   - Calculate and enhance histograms for contrast improvement.

5. **Color Space Transformations**:
   - Convert images between BGR and YUV, and apply enhancements.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/image-processing-python.git
   cd image-processing-python
   ```

2. Place your input image in the project directory.

3. Run the scripts section by section to explore the various features. For example:
   ```bash
   python basic_operations.py
   ```

4. Save the results to view transformed images.

## Examples
### Rotating an Image
```python
# Rotate the image by 90 degrees
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
```
### Applying CLAHE
```python
# Apply CLAHE to enhance local contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)
```
### Augmentation with imgaug
```python
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Rotate((-45, 45)),
    iaa.AdditiveGaussianNoise(scale=(10, 60))
])
images_aug = seq(images=[image])
```
### Histogram Equalization in YUV
```python
# Convert to YUV and equalize the Y channel
yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you want to improve this project.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

