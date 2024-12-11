# Handwritten Digit Recognition
### This project implements a simple handwritten digit recognition system using Python. Users can draw digits on a canvas, and the system predicts the digit based on the MNIST dataset. The project demonstrates a basic machine learning workflow using the K-Nearest Neighbors (KNN) algorithm, with options to improve performance using preprocessing techniques, data augmentation, and transitioning to a Convolutional Neural Network (CNN).

## Features
- Interactive Canvas: Users can draw digits on a GUI canvas.
- Real-Time Prediction: Predicts the digit drawn on the canvas.
- MNIST Dataset Integration: Trained on the MNIST dataset of handwritten digits.
- Preprocessing: Includes advanced preprocessing steps like centering, resizing, and normalizing input digits.
- Expandable: Optionally supports CNNs for better accuracy.

## Prerequisites
Ensure you have the following installed:

- Python 3.8 or higher

You can install the dependencies using the following command (windows): 

pip install -r requirements.txt 

## Usage
### Step 1: Run the Script
Execute the Python script to launch the GUI:
python digit_recognition.py
### Step 2: Draw a Digit
Use your mouse to draw a digit on the canvas.
Ensure the digit is clear and fully within the canvas boundaries.
### Step 3: Predict the Digit
Click the Predict button to see the predicted digit.
The result will be displayed below the canvas.
### Step 4: Clear the Canvas
Click the Clear button to reset the canvas and draw a new digit.

