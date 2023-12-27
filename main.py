import tkinter as tk
from tkinter import Label, Text, filedialog
from PIL import Image, ImageTk, ImageDraw
import pytesseract
from predict import ObjectDetection
import tensorflow as tf
import numpy as np
import os
import re

pytesseract.pytesseract.tesseract_cmd = r"./Tesseract-OCR/tesseract.exe"
MODEL_FILENAME = 'saved_model.pb'
LABELS_FILENAME = 'labels.txt'

class TFObjectDetection(ObjectDetection):
    def __init__(self, model_filename, labels):
        super(TFObjectDetection, self).__init__(labels)
        model = tf.saved_model.load(os.path.dirname(model_filename))
        self.serve = model.signatures['serving_default']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR
        inputs = tf.convert_to_tensor(inputs)
        outputs = self.serve(inputs)
        return np.array(outputs['outputs'][0])

def get_absolute_bounding_box(image, relative_bounding_box):
    left = int(relative_bounding_box['left'] * image.width)
    top = int(relative_bounding_box['top'] * image.height)
    right = int((relative_bounding_box['left'] + relative_bounding_box['width']) * image.width)
    bottom = int((relative_bounding_box['top'] + relative_bounding_box['height']) * image.height)
    left = max(0, left)
    top = max(0, top)
    right = min(image.width, right)
    bottom = min(image.height, bottom)

    return (left, top, right, bottom)

def load_image_and_detect():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        with open(LABELS_FILENAME, 'r') as f:
            labels = [label.strip() for label in f.readlines()]

        od_model = TFObjectDetection(MODEL_FILENAME, labels)
        image = Image.open(file_path)

        # Step 1: Detect license plate region
        plate_predictions = od_model.predict_image(image)

        # Display either the original image with the bounding box or show a message if license plate not found
        if not display_highest_probability_bounding_box(image, plate_predictions):
            display_license_plate_not_found()

def perform_ocr(image, bounding_box):
    # Crop the license plate region
    plate_region = image.crop(bounding_box)

    # Convert the cropped image to grayscale
    gray_plate = plate_region.convert("L")

    # Use pytesseract to perform OCR
    plate_text = pytesseract.image_to_string(gray_plate, config='--psm 6')  # Use --psm 6 for sparse text

    # Filter out non-alphanumeric characters (keep only uppercase letters and numbers)
    plate_text_filtered = re.sub(r'[^A-HJ-NP-Z0-9]', '', plate_text)

    return plate_text_filtered.strip()  # Remove leading/trailing whitespaces

def display_highest_probability_bounding_box(image, plate_predictions):
    highest_probability_index = -1
    highest_probability = -1

    # Find the index of the highest-probability 'carplate' prediction
    for i, plate_prediction in enumerate(plate_predictions):
        if plate_prediction['tagName'] == 'carplate' and plate_prediction['probability'] > highest_probability:
            highest_probability = plate_prediction['probability']
            highest_probability_index = i

    # Display either the original image with the bounding box or return False if license plate not found
    if highest_probability_index != -1:
        plate_prediction = plate_predictions[highest_probability_index]
        bounding_box = get_absolute_bounding_box(image, plate_prediction['boundingBox'])

        # Perform OCR on the highest-probability 'carplate' bounding box
        ocr_result = perform_ocr(image, bounding_box)

        # Clear previous OCR results
        results_text.config(state="normal")
        results_text.delete(1.0, "end")

        # Display the new OCR result in the Text widget
        results_text.insert("end", f"OCR Result: {ocr_result}\n")
        results_text.config(state="disabled")

        # Draw the bounding box directly on the original image
        draw = ImageDraw.Draw(image)
        draw.rectangle(bounding_box, outline="red", width=2)

        # Resize the image for display
        resized_image = image.resize((400, 400))  # Adjust the size as needed

        # Update the image on the GUI
        image_tk = ImageTk.PhotoImage(resized_image)
        image_label.configure(image=image_tk)
        image_label.image = image_tk

        # Display the cropped image on the right side
        cropped_image = image.crop(bounding_box)
        resized_cropped_image = cropped_image.resize((400, 400))  # Adjust the size as needed
        cropped_image_tk = ImageTk.PhotoImage(resized_cropped_image)
        cropped_image_label.configure(image=cropped_image_tk)
        cropped_image_label.image = cropped_image_tk

        return True
    else:
        return False

def display_license_plate_not_found():
    # Clear previous OCR results
    results_text.config(state="normal")
    results_text.delete(1.0, "end")

    # Display the message in the Text widget
    results_text.insert("end", "License plate cannot be found.\n")
    results_text.config(state="disabled")

if __name__ == '__main__':
    root = tk.Tk()
    root.title("License Plate Detection")
    root.geometry("1200x900")  # Increased width for both images

    image_label = Label(root)
    image_label.grid(row=0, column=0, padx=5, pady=5)

    cropped_image_label = Label(root)
    cropped_image_label.grid(row=0, column=1, padx=5, pady=5)

    results_text = Text(root, height=10, state="disabled")
    results_text.grid(row=3, column=0, padx=10, pady=10, columnspan=2)

    load_button = tk.Button(root, text="Select Image", command=load_image_and_detect)
    load_button.grid(row=4, column=0, pady=10, padx=10, columnspan=2)

    root.mainloop()
