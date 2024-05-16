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
![0](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/227ad594-8ee4-423e-bd88-eccbbb63fa03)

![1_0](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/282beac5-3aa9-4e31-bcc6-293cd317ffd5)

![1_1](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/d3fcf778-972e-405a-a25c-ad9f2d03bc87)

![1_2](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/5250a0a6-ff10-42ca-852d-fd958dc1ef6a)

![2_0](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/67d00a3a-bbf9-413f-ad28-69c88c6f6567)

![2_1](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/0025c313-fcb4-4f25-adfe-859c7921e48c)

![2_2](https://github.com/siddartharc/Scriptify-Tool-for-Telugu-Character-Detection/assets/83510588/62edd463-3a1a-482b-bb9b-0031e619a7b4)



