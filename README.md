Here's an improved version of the instructions you provided for your tool's README file:

---

# Handwritten Image to Text Converter Tool

This tool facilitates the conversion of handwritten images to text files using various options and configurations.

## Features

1. **Image Preprocessing using Cosine Similarity:**
   - This option preprocesses the images using cosine similarity.
   - To change the image size during preprocessing, modify the image size parameter in the `preprocessing.py` file at line 26.
   - Upload images for preprocessing to `\Tool\data\image\input`.

2. **Model Training with Custom Data:**
   - Train the model using your own data.
   - Adjust the number of epochs for hyperparameter tuning in the `model_optuna.py` file at line 14.
   - Specify the number of trials for parameter variation by changing the number of trials in line 104 of the same file.
   - Upload training data to `\Tool\data\training_data`.

3. **Text Image to Text File Conversion:**
   - Convert text images to text files.
   - Enter the threshold for character detection and set the step size based on the distance between lines in your text file.
   - Find the converted text files in `\Tool\data\image\output`.

## Usage

1. Clone or download the repository to your local machine.
2. Install the necessary dependencies.
3. Follow the specific instructions for each option:
   - For image preprocessing, upload images to `\Tool\data\image\input` and adjust settings in `preprocessing.py`.
   - For model training, provide your training data in `\Tool\data\training_data` and configure parameters in `model_optuna.py`.
   - For text image conversion, enter threshold and step size parameters as required.
4. Execute the tool and access the output files accordingly.

## Output Locations

- Preprocessed images: `\Tool\data\image\output`
- Training data: `C:\Users\sidda\Desktop\Tool\data\preprocessed_data`
- Converted text files: `\Tool\data\image\output`

---
![Screenshot 2024-05-16 132347](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/c2c7d801-c09b-44cf-a573-9256fbf07d03)
![Screenshot 2024-05-16 132209](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/e9acc3e0-ff3d-49b2-b6fe-9ace92005f69)
![Screenshot 2024-05-16 132123](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/691e56bf-b63b-4092-bff1-06e440c51963)
![Screenshot 2024-05-16 132102](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/48ffa929-7690-4f91-a72c-31bb579d3435)
![Screenshot 2024-05-16 132018](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/f8712a74-ab2b-463f-9e28-ab3cddb8cfc5)
![Screenshot 2024-05-16 131952](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/29dc5e00-4973-4c4a-b3f7-851fb3c26d92)
![Screenshot 2024-05-16 132406](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/73b56624-0df1-44eb-aaf9-89b3bb8c4d35)


