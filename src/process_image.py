import cv2
from PIL import Image
import numpy as np

def process_leaf_image(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate the average brightness of the image
    avg_brightness = np.mean(gray_img)
    
    # Decide on processing based on the average brightness
    if avg_brightness < 128:
        # If the image is dark, it might be inverted depending on your needs
        # For this example, we'll assume dark images are mostly fine as is
        processed_img = gray_img
    else:
        # If the image is light, invert it to make the leaf darker than the background
        processed_img = 255 - gray_img
    
    # Apply a binary threshold to enhance contrast
    # Adjust the threshold value (120) as needed for your images
    _, threshold_img = cv2.threshold(processed_img, 120, 255, cv2.THRESH_BINARY)
    
    # Save the processed image
    cv2.imwrite(output_path, threshold_img)
    
    # Optionally, display the image
    cv2.imshow('Processed Leaf Image', threshold_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
process_leaf_image('data/leaf_photos/Quercus_Coccifera.jpeg', 'data/leaf_photos/Quercus_Coccifera_processed4.jpeg')
