import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import preprocess_region
# cnn_path = r"C:\Users\sidda\Desktop\RCS\OCR\Model\the_best_model.h5"
# loaded_model = load_model(cnn_path)
# output_path = r'C:\Users\sidda\Desktop\Tool\data\image\output'

base_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(base_dir) 
cnn_path = os.path.join(base_dir, "models", "CNN.h5")
loaded_model = load_model(cnn_path)
output_path = os.path.join(base_dir, "data", "image", "output")


def predict_character(image, model, threshold=0.5):
    telugu_letters = ["అ", "ఆ", "ఇ", "ఈ", "ఉ", "ఊ", "ఋ", "ౠ", "ఎ", "ఏ", "ఐ", "ఒ", "ఓ", "ఔ",
                  "అం", "అః", "క", "ఖ", "గ", "ఘ", "ఙ", "చ", "ఛ", "జ", "ఝ", "ఞ", "ట",
                  "ఠ", "డ", "ఢ", "ణ", "త", "థ", "ద", "ధ", "న", "ప", "ఫ", "బ", "భ", "మ",
                  "య", "ర", "ల", "వ", "శ", "ష", "స", "హ", "ళ", "క్ష", "ఱ"]
    resized_image = cv2.resize(image, (255, 255), interpolation=cv2.INTER_LINEAR)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image_input = np.expand_dims(rgb_image, axis=0)
    prediction = model.predict(image_input)
    predicted_class = np.argmax(prediction)
    
    # Check if the prediction confidence is above the threshold
    if prediction[0][predicted_class] > threshold:
        return telugu_letters[predicted_class]
    else:
        return None  # Return None if confidence is below threshold
def renumber_boxes(line_clusters,boxes_coords):
    merged_boxes = []
    merged_numbers = []
    for line_idx, sub_cluster in enumerate(line_clusters):
        # Merge overlapping bounding boxes
        merged_boxes_line = []
        merged_numbers_line = []
        for idx, point_idx in enumerate(sub_cluster):
            x1, y1, x2, y2 = boxes_coords[point_idx]
            merged_boxes_line.append((x1, y1, x2, y2))
            merged_numbers_line.append([f'{line_idx + 1}.{idx+1}'])

        merged_boxes.extend(merged_boxes_line)
        merged_numbers.extend(merged_numbers_line)
    return merged_boxes,merged_numbers
def sorting(merged_boxes,merged_numbers,i):
    l = 1
    lines = []  # Initialize a list to store lines of characters
    current_line = []  # Initialize a list for the current line
    # Plot the merged boxes on the image
    image = cv2.imread(i)
    for box, numbers in zip(merged_boxes, merged_numbers):
        x1, y1, x2, y2 = box
        region_image = image[int(y1):int(y2), int(x1):int(x2)]
            
            # Preprocess the region
        preprocessed_region = preprocess_region(region_image)
            
            # Predict the character
        predicted_character = predict_character(preprocessed_region, loaded_model)
        if(predicted_character!=None):
            box_number = int(numbers[0].split('.')[0])

            if box_number == l:  # If the box number is equal to the current line number
                current_line.append(predicted_character)  # Add the character to the current line
            else:  # If the box number changes
                lines.append(current_line)  # Add the current line to the lines list
                current_line = [predicted_character]  # Start a new line with the current character
                l = box_number  # Update the current line number
        else:
            continue
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, ', '.join(numbers), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Add the last line to the lines list
    lines.append(current_line)
    image_name = os.path.basename(i).split('.')[0]
    output_file_path = os.path.join(output_path, f"{image_name}.txt")

    # Write the lines to the output file
    with open(output_file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(' '.join(line) + '\n')
    f.close()
    print("Lines written to:", output_file_path)
    output_image_path = os.path.join(output_path,image_name)
    cv2.imwrite(f"{output_image_path}.png", image)
    