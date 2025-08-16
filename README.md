# CAPTCHA-Recognition-using-CNN
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)  ![CNN](https://img.shields.io/badge/CNN-DeepLearning-blue?style=for-the-badge)  ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-AI-green?style=for-the-badge)  

## Introduction
CAPTCHAs—short for Completely Automated Public Turing test to tell Computers and Humans Apart—are widely used security mechanisms designed to distinguish human users from automated bots. Typically presented as distorted text embedded within an image, CAPTCHAs require users to manually transcribe the characters before gaining access to a website or completing a form. This process serves as a safeguard against automated abuse, such as spam submissions or credential stuffing.

However, advancements in Deep Learning and Computer Vision have challenged the efficacy of traditional CAPTCHA systems. This project explores the application of Convolutional Neural Networks (CNNs) to automatically interpret and solve text-based CAPTCHAs, effectively bypassing their intended security function.

CNNs are a specialized class of deep learning architectures that excel in image processing tasks. By learning hierarchical feature representations, CNNs can extract meaningful patterns from visual data, making them particularly well-suited for tasks such as character recognition in distorted or noisy images. This capability positions CNNs as a powerful tool in the domain of CAPTCHA decoding.

This project aims to defeat text-based CAPTCHA challenges by training a Convolutional Neural Network (CNN) to automatically recognize and predict CAPTCHA text.

### The workflow involves:
1. *Dataset preparation & preprocessing*
2. *CNN model architecture design*
3. *Training & evaluation on CAPTCHA samples*
4. *Prediction & testing on unseen CAPTCHA images*

## Dataset Description
1. *Dataset:* 1070 PNG images of text-based CAPTCHAs
2. *Source:* ResearchGate CAPTCHA Dataset
3. Each image contains *5-character strings*
4. *Character set:* 26 lowercase English letters + digits (0–9)
5. *Split:* 970 images for training, 100 images for testing
6. *Image size:* 50×200 pixels

## Model Development
The model is based on a 24-layer Convolutional Neural Network (CNN)

🔄 Workflow
1. Data Preprocessing
* &nbsp;Convert images to grayscale to reduce noise
* &nbsp;Normalize and reshape images to (50, 200, 1)
* &nbsp;Extract labels from filenames (since filenames contain the CAPTCHA text)
* &nbsp;Encode labels into a 5×36 one-hot array (5 characters × 36 possible classes)

2. Model Architecture
* &nbsp;Input Layer: Grayscale image (50×200×1).
* &nbsp;Convolution + MaxPooling Layers: Extract features.
* &nbsp;Batch Normalization Layer: Improves training stability.
* &nbsp;Flatten Layer: Converts pooled features into a 1D vector.
* &nbsp;Dense Layers (ReLU activation): Learn feature patterns.
* &nbsp;Dropout Layers: Prevent overfitting.
* &nbsp;Output Dense Layers (Sigmoid activation): Predict 5 characters independently.
* &nbsp;Optimizer: Adam
* &nbsp;Loss Function: Categorical Cross-Entropy

## Installation & Setup
Follow these steps to set up and run the CAPTCHA Recognition project:

*1. Clone the repository or download ZIP*
* *Option A*
<br>&nbsp;&nbsp;&nbsp;git clone https://github.com/your-username/CAPTCHA-Recognition-using-CNN.git
<br>&nbsp;&nbsp;&nbsp;cd CAPTCHA-Recognition-using-CNN
* *Option B*
<br>&nbsp;&nbsp;&nbsp;Download the repository as a ZIP file and extract it.

*2. Download the Dataset*
* Download the CAPTCHA dataset from ResearchGate CAPTCHA Dataset
* The dataset contains 1070 images of 5-character CAPTCHAs

*3. Add Dataset to Project*
* Place the dataset folder in your project directory (or upload to Google Drive if using Colab)
* Ensure that the notebook or scripts point to the correct dataset path

*4. Install Dependencies*
=> Install Python packages required for the project:
<br>&nbsp;&nbsp;&nbsp;pip install -r requirements.txt

*5. Execute the Notebook*
* Open Captcha_Recognition.ipynb in Jupyter Notebook or Google Colab
* Run the cells sequentially:
<br>1️⃣ Data preprocessing – load and process images
<br>2️⃣ Model building – define CNN architecture
<br>3️⃣ Training – train the CNN on training images
<br>4️⃣ Evaluation – test the model on unseen CAPTCHA images
<br>5️⃣ Inference – predict new CAPTCHA samples

*6. View Results*
<br>&nbsp;&nbsp;After training, the model outputs:
* Training loss and accuracy graphs
* Validation/test accuracy for each character
* Predicted CAPTCHA text for sample images
  
## System Design
1. Data Preparation
2. Image Preprocessing - Grayscale Conversion, Binarization, Noise Removal or Segmentation
3. Model Design and Training - Convolution Neural Network

<p align="center">
<img width="570" height="500" alt="image" src="https://github.com/user-attachments/assets/a34a6366-174c-41eb-bf7d-05b9a347a27f" />
<br><br>
<img width="918" height="307" alt="Screenshot 2024-12-18 220837 (1)" src="https://github.com/user-attachments/assets/7b78dde9-9196-4472-a432-cc1e3d327bae" />
</p>

## Implementaion - Convolution Neural Network Structure
<p align="center"><img width="800" height="600" alt="image (4)" src="https://github.com/user-attachments/assets/859f9503-b8f6-4cce-b4eb-930c0bf5ceb9" /></p>

## Model Architecture
The CAPTCHA recognition model is a 24-layer Convolutional Neural Network (CNN) designed to predict each character in a 5-character CAPTCHA independently.

### 🔹 Overview of Layers
1. *Input Layer* => Accepts grayscale images of size 50×200×1
2. *Convolutional Layers* => Extracts spatial features from CAPTCHA images and uses multiple filters to detect edges, curves, and patterns
3. *Max Pooling Layers* => Reduces spatial dimensions while keeping important features
4. *Batch Normalization Layer* => Improves model stability and speeds up training
5. *Flatten Layer* => Converts 2D feature maps into a 1D vector for fully connected layers
6. *Dense (Fully Connected) Layers* => Five dense layers connected to flatten output where each dense layer uses ReLU activation
7. *Dropout Layers* => Connected after each dense layer for regularization to prevent overfitting
8. *Output Layers* => Five independent dense layers (one for each character) where each output layer has 36 neurons (26 letters + 10 digits). Uses Sigmoid activation to predict probabilities of each character class.

### 🔹 Training Details
* *Optimizer*: Adam
* *Loss Function*: Categorical Cross-Entropy
* *Total Parameters*: 1,818,196 => *Trainable: 1,818,132 and Non-trainable: 64*
  
### 🔹 Input & Output
* *Input:* Grayscale image (50, 200, 1)
* *Output:* Five probability vectors (5 × 36) corresponding to each character in the CAPTCHA
  
## Results 
<p align="center">
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/b7df53fe-aa09-4cc8-822f-76089751ea51" />
<br>
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/6f76ffcd-8664-4553-b6d9-6c6520893df4" />
<br>
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/1058d6d3-b1ac-4961-ba9f-349bc2a7d567" />
</p>

## ✅ Conclusion
* CAPTCHA was designed to prevent bots, but Deep Learning algorithms like CNNs can defeat this defense.
* A 24-layer CNN was trained on 1070 CAPTCHA images (5 characters each).
* The model achieved high accuracy across individual characters and can predict 5-letter CAPTCHAs reliably.
* *Future scope*: Extend this work to more complex CAPTCHAs containing symbols, noise, and distortion and Implement Recurrent Neural Networks (RNN/LSTM) for sequential character recognition

## 👤 Author
- Asiya Anjum S
- GitHub: AsiyaAnju
- LinkedIn: https://www.linkedin.com/in/asiyaanjums
