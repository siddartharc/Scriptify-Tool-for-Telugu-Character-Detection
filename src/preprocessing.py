import os
from tqdm import tqdm
import cv2
from Preprocessing_func import compute_cosine_distance, draw_cosine_similarity_maps, calculate_kernel_size, calculate_otsu_threshold
import numpy as np

def load_data_pp(train_dir):
    # preprocessed_dir = r'C:\Users\sidda\Desktop\Tool\data\Preprocessed_data'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base_dir) 
    preprocessed_dir = os.path.join(base_dir, 'data', 'Preprocessed_data')
    distance_maps_dict = {}
    
    # Walk through the directories and process images
    for root, dirs, files in os.walk(train_dir):
        # Get the relative path from the train_dir
        relative_path = os.path.relpath(root, train_dir)
        
        # Create the corresponding output directory structure in preprocessed_dir
        output_dir = os.path.join(preprocessed_dir, relative_path)
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in tqdm(files, desc="Processing images"):
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)
            
            # Preprocess the image
            kernel_size = calculate_kernel_size(image)
            kernel = np.ones((1, 1), np.uint8)
            thickened_image = cv2.dilate(image, kernel, iterations=1)
            otsu_thresholded_image = calculate_otsu_threshold(thickened_image)
            smoothed_image = cv2.GaussianBlur(thickened_image, (kernel_size, kernel_size), 0)
            
            # Calculate dynamic reference points based on image size
            height, width = thickened_image.shape
            mid_height = height // 2
            mid_width = width // 2
            tenth_height = height // 10
            tenth_width = width // 10

            reference_points = [
                thickened_image[tenth_height, tenth_width],               # Top-left corner
                thickened_image[tenth_height, width - tenth_width - 1],    # Top-right corner
                thickened_image[height - tenth_height - 1, tenth_width],   # Bottom-left corner
                thickened_image[height - tenth_height - 1, width - tenth_width - 1],  # Bottom-right corner
                thickened_image[mid_height, tenth_width],                  # Middle-top
                thickened_image[tenth_height, mid_width],                  # Middle-left
                thickened_image[tenth_height, width - mid_width - 1],      # Middle-right
                thickened_image[height - tenth_height - 1, mid_width]      # Middle-bottom
            ]

            similarity_maps = draw_cosine_similarity_maps(otsu_thresholded_image, reference_points)

            # Combine the similarity maps
            intersection = np.minimum.reduce(similarity_maps)
            intersection = cv2.resize(intersection, (255, 255), interpolation=cv2.INTER_LINEAR)
            
            # Save the preprocessed image in the output directory
            output_filename = os.path.splitext(filename)[0] + "_preprocessed.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, intersection)
            
def preprocess_region(region):
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    resized_region = cv2.resize(gray_region, (124, 124), interpolation=cv2.INTER_LINEAR)
    kernel = np.ones((1, 1), np.uint8)
    thickened_region = cv2.dilate(resized_region, kernel, iterations=1)
    _, thresholded_region = cv2.threshold(thickened_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_region