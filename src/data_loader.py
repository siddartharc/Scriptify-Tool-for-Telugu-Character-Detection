import os
from tqdm import tqdm
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

def data_loader(data_dir,num_images_per_class):
    # Initialize lists to store images and labels
    # Number of images per class
    # num_images_per_class = 100
    datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
    )
    folder_names = sorted(os.listdir(data_dir))  # Ensure folders are sorted D1, D2, ..., D52

    total_progress_bar = tqdm(folder_names, desc="Processing Folders", position=0)

    images = []
    labels = []

    for folder_name in total_progress_bar:
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            images_in_folder = os.listdir(folder_path)
            num_images = min(num_images_per_class, len(images_in_folder))

            loaded_images = []
            for image_name in images_in_folder[:num_images]:
                try:
                    image_path = os.path.join(folder_path, image_name)
                    loaded_image = cv2.imread(image_path)
                    if loaded_image is None:
                        print(f"Skipping image: {image_name} in folder {folder_name}, image loading failed.")
                        continue

                    loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)

                    # Resize the image to size 255x255
                    loaded_image = cv2.resize(loaded_image, (255, 255))

                    loaded_images.append(loaded_image)
                except Exception as e:
                    print(f"Error loading image: {image_name} in folder {folder_name}, Error: {e}")
                    continue

            try:
                label = int(''.join(filter(str.isdigit, folder_name)))
                labels.extend([label] * len(loaded_images))
            except ValueError:
                print(f"Invalid folder name: {folder_name}, cannot extract numeric part.")
                continue

            images.extend(loaded_images)
    return images,labels
def data_model(images,labels):

    unique_labels = set(labels)
    print("Unique Labels:", unique_labels)

    num_classes = len(unique_labels)

    labels = [label - 1 for label in labels]

    one_hot_labels = to_categorical(labels, num_classes=num_classes)

    print("Number of classes:", num_classes)

    X_train, X_temp, y_train, y_temp = train_test_split(images, one_hot_labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Shuffle training and validation data
    train_data = list(zip(X_train, y_train))
    np.random.shuffle(train_data)
    X_train, y_train = zip(*train_data)

    val_data = list(zip(X_val, y_val))
    np.random.shuffle(val_data)
    X_val, y_val = zip(*val_data)
    # Convert data to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return num_classes,X_train,y_train,X_val,y_val,X_test,y_test
def get_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):  # Add more file extensions if needed
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    return image_paths