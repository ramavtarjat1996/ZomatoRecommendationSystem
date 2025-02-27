Automatic Number Plate Recognition (ANPR)
Overview
This project implements an Automatic Number Plate Recognition (ANPR) system that can detect and recognize vehicle license plates from images using OpenCV and Tesseract OCR. The system processes vehicle images, detects the license plate, and extracts the plate number.

The system performs the following steps:

Image Preprocessing: Convert the image to grayscale, apply Gaussian blur, and use Canny edge detection.
License Plate Detection: Detect contours in the image and find the rectangular shape of the license plate.
Optical Character Recognition (OCR): Use Tesseract OCR to recognize the text on the license plate.
Features
Vehicle License Plate Detection: Detects the license plate area within a vehicle image.
OCR for Text Recognition: Uses Tesseract OCR to extract the vehicle's license plate number.
Edge Detection & Preprocessing: Utilizes image processing techniques to prepare the image for license plate recognition.
Real-time Processing: The system can be extended to work on video streams (real-time vehicle license plate recognition).
Requirements
To run this project, you will need the following Python libraries:

Python 3.x
OpenCV
pytesseract
numpy
Dependencies
You can install the required libraries using the requirements.txt file:

bash
Copy
pip install -r requirements.txt
Additionally, you need to install Tesseract OCR on your machine:

Windows: Download from Tesseract OCR for Windows.
macOS: Install with Homebrew:
bash
Copy
brew install tesseract
Linux:
bash
Copy
sudo apt-get install tesseract-ocr
Tesseract Configuration (Windows)
If you're using Windows, make sure to specify the path to the Tesseract OCR executable in the script:

python
Copy
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this with the actual path
How It Works
Image Preprocessing:

The system first converts the input image to grayscale and applies Gaussian blur to reduce noise.
Canny edge detection is then used to find the edges of the objects in the image.
License Plate Detection:

The system uses contour detection to find potential areas in the image where the license plate might be.
Based on the shape and aspect ratio, the system filters out non-license plate objects and locates the plate.
OCR for License Plate Recognition:

Once the license plate is detected, Tesseract OCR is used to extract the license plate number from the image.
Setup
Clone the repository:

bash
Copy
git clone <your-repository-url>
Install the required dependencies:

bash
Copy
pip install -r requirements.txt
Place your vehicle images inside the data/ directory (e.g., vehicle_image_1.jpg, vehicle_image_2.jpg).

Run the system with the following command:

bash
Copy
python src/anpr.py
Usage
Place Image: Add your vehicle image inside the data/ folder (e.g., vehicle_image_1.jpg).

Run the Script: In your terminal, navigate to the src/ folder and run:

bash
Copy
python anpr.py
Output:

The system will display the input image with a bounding box around the detected license plate.
The recognized license plate number will be printed on the terminal.
Example Output:
For the input image vehicle_image_1.jpg, the output might look like:

yaml
Copy
Detected Number Plate: ABC 1234
Folder Structure
perl
Copy
anpr-system/
│
├── data/                           # Folder for images (e.g., vehicle images)
│   └── vehicle_image_1.jpg         # Sample vehicle image
│   └── vehicle_image_2.jpg         # Another vehicle image
│
├── src/                            # Folder for source code
│   └── anpr.py                     # Main code for ANPR (license plate recognition)
│
├── requirements.txt                # List of required Python packages
│
├── README.md                       # Documentation for the project
│
└── LICENSE #   Z o m a t o R e c o m m e n d a t i o n S y s t e m  
 #   Z o m a t o R e c o m m e n d a t i o n S y s t e m  
 #   Z o m a t o R e c o m m e n d a t i o n S y s t e m  
 #   A u t o m a t i c _ N u m b e r _ P l a t e _ R e c o g n i t i o n  
 