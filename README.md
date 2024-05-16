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
