import cv2
import pytesseract
import numpy as np

# Path to the Tesseract executable (Windows only)
# If you're using Windows, make sure Tesseract is installed and provide the path.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Change this path if needed

def preprocess_image(image_path):
    """Preprocess the image for license plate detection."""
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 30, 150)

    return img, edges

def detect_plate(edges):
    """Detect the number plate in the image."""
    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area to find the largest contour (likely the number plate)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        # Get the bounding box of each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Define the aspect ratio for a license plate (you can tune this threshold)
        aspect_ratio = w / float(h)
        
        if aspect_ratio >= 2 and aspect_ratio <= 6:  # Aspect ratio of number plates
            plate = edges[y:y+h, x:x+w]
            return plate, (x, y, w, h)

    return None, None

def ocr_recognition(plate_image):
    """Recognize the text on the plate using Tesseract OCR."""
    # Use Tesseract to extract text
    text = pytesseract.image_to_string(plate_image, config='--psm 8')
    return text.strip()

def anpr(image_path):
    """Main function for Automatic Number Plate Recognition."""
    # Preprocess the image
    img, edges = preprocess_image(image_path)
    
    # Detect number plate
    plate, plate_bbox = detect_plate(edges)
    
    if plate is not None:
        # OCR to extract text
        plate_text = ocr_recognition(plate)
        
        # Draw the bounding box around the detected plate
        x, y, w, h = plate_bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow("Detected Plate", img)
        print(f"Detected Number Plate: {plate_text}")
        
        # Wait until any key is pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No number plate detected in the image.")

if __name__ == '__main__':
    
    image_path = 'data/vehicle_image_1.jpg' 
    anpr(image_path)
