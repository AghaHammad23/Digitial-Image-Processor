# Image Processing Project - README

This project provides an interactive GUI application for performing various image processing tasks, including Affine Transformations, Logical Operations, Edge Detection, Image Arithmetic, and more. The application is built with Python using Tkinter for the interface and OpenCV for image processing.

## Features

### 1. **Affine Transformations**
   - **Translation**: Move the image horizontally or vertically.
   - **Rotation**: Rotate the image around its center by a specified angle.
   - **Scaling**: Resize the image by increasing or decreasing its dimensions.
   - Adjustable sliders allow precise control over the transformation parameters.

### 2. **Logical Operations**
   - **AND**: Perform a bitwise AND operation between two images.
   - **OR**: Perform a bitwise OR operation between two images.
   - **XOR**: Perform a bitwise XOR operation between two images.
   - **NOT**: Perform a bitwise NOT operation on a single image.

### 3. **Edge Detection**
   - **Sobel**: Detect edges using Sobel operator with adjustable kernel size.
   - **Canny**: Detect edges using Canny edge detection with adjustable thresholds.
   - **Prewitt**: Use Prewitt operator for edge detection.

### 4. **Image Arithmetic**
   - **Addition**: Add two images together.
   - **Subtraction**: Subtract one image from another.
   - **Multiplication**: Multiply two images pixel-wise.
   - **Division**: Divide one image by another pixel-wise.
   - **Blending**: Blend two images with adjustable alpha value.

### 5. **Image Transforms**
   - **Fourier Transform**: Compute the frequency spectrum of an image.
   - **Wavelet Transform**: Perform wavelet decomposition for feature extraction.
   - **Hough Transform**: Detect lines in an image using the Hough Transform.

### 6. **Face Recognition**
   - **Detect Faces**: Locate faces in an image using Haar cascades.
   - **Recognize Faces**: Advanced face detection with recognition markers and eye detection.

## Installation

1. **Install Dependencies**
   Make sure you have Python installed, then install the required libraries:
   ```bash
   pip install opencv-python opencv-python-headless pillow numpy pywavelets
   ```

2. **Run the Application**
   Execute the following command to launch the GUI:
   ```bash
   python main.py
   ```

## Usage

1. **Load Images**
   - Click on the "Load Primary Image" or "Load Secondary Image" buttons to load input images.
   - Ensure that the images have compatible dimensions for operations requiring two inputs.

2. **Select Category and Operation**
   - Use the dropdown menus to select an image processing category and operation.

3. **Adjust Parameters**
   - For operations with adjustable parameters (e.g., translation, scaling, rotation, or Canny edge detection), use the provided sliders.

4. **Process and Save**
   - Click "Process" to execute the selected operation.
   - Use "Save Result" to save the processed image.


## Troubleshooting

- **Error: Images must be the same size**
  Ensure both images have the same dimensions for operations requiring two inputs.

- **Error: Failed to load image**
  Make sure the file format is supported (.jpg, .jpeg, .png, .bmp, .tiff).

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code.

## Credits
- **OpenCV**: For providing powerful image processing tools.
- **Tkinter**: For the graphical user interface.
- **PyWavelets**: For wavelet transform operations.

