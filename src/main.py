import warnings
warnings.filterwarnings("ignore")
from preprocessing import load_data_pp
from train import train
from data_loader import get_image_paths
from text_detection import text_detection
from Clustering import Clustering
from predict import renumber_boxes, sorting
import os
from scriptify import  print_scriptify
print_scriptify()
print("Enter 1 to preprocess the data")
print("Enter 2 to train the model")
print("Enter 3 to convert image to text")
user_input = int(input("Enter your input here: "))

# Assuming your script is run from the same directory where these folders are located
base_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(base_dir) 
training_dir = os.path.join(base_dir, 'data', 'training_data')
input_folder = os.path.join(base_dir, 'data', 'image', 'input')

if user_input == 1:
    print("Ensure that the files are present at 'data/training_data'")
    print("This should be the way of data arrangement:")
    print("""
    |
    |--- class 1 folder (images)
    |--- class 2 folder (images)
    |--- and so on
    """)
    load_data_pp(train_dir=training_dir)

elif user_input == 2:
    print("Ensure that preprocessing is done first")
    epochs = int(input("Enter the number of epochs: "))
    num_images = int(input("Enter the number of images per class: "))
    train(epochs, num_images)
    print("Your model is saved at 'models/trained_model.h5'")

elif user_input == 3:
    print(f"Ensure that the images are present at {input_folder}")
    image_paths = get_image_paths(input_folder)
    for i in image_paths:
        results = text_detection(i)
        line_clusters, boxes_coords = Clustering(results)
        boxes, numbers = renumber_boxes(line_clusters, boxes_coords)
        sorting(boxes, numbers, i)
    print("In the box numbers, a.b indicates the bth character in the ath line.")