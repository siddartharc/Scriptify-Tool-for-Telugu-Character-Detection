import numpy as np
import cv2
def compute_cosine_distance(reference_feature, pixel_feature):
    cosine_sim = np.dot(reference_feature, pixel_feature) / (np.linalg.norm(reference_feature) * np.linalg.norm(pixel_feature))
    return 1 - cosine_sim 

def draw_cosine_similarity_maps(image, reference_points):
    height, width = image.shape 
    similarity_maps = []

    for reference_point in reference_points:
        similarity_map = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                pixel_feature = image[y, x]
                similarity = compute_cosine_distance(reference_point, pixel_feature)
                similarity_map[y, x] = similarity

        similarity_map = (similarity_map * 255).astype(np.uint8)
        similarity_maps.append(similarity_map)

    return similarity_maps

def calculate_otsu_threshold(image):
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_image

def calculate_kernel_size(image):
    height, width = image.shape
    avg_size = (height + width) // 2
    kernel_size = avg_size // 30  
    # Ensure kernel size is an odd number greater than zero
    kernel_size = max(kernel_size, 1)  # Ensure it's at least 1
    kernel_size += 1 if kernel_size % 2 == 0 else 0  # Ensure it's odd
    return kernel_size
