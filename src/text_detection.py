import os
import cv2
from ultralytics import YOLO
from IPython.display import display, Image
from sklearn.metrics import pairwise_distances
# from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from plot_boxes import plot

def text_detection(i):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base_dir) 
    yolo_path = os.path.join(base_dir, 'models', 'yolo.pt')
    model = YOLO(yolo_path)
    while True:
        threshold = int(input('enter the threshold btw 1 to 100 and check if all the characters were being detected : ')) /100
        image = cv2.imread(i)
        results = model(image)[0]
        r = plot(results,threshold,image,model)
        # print(r)
        user_input = input("Is most of the text Detected ? (Y/N): ")
        if user_input.lower() == 'y':
            return r
            break
