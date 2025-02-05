# Potato Leaf Disease Detection

## Overview
Potato leaf diseases can severely impact crop yields and food security. This project leverages **deep learning** to detect **potato leaf diseases** using **Convolutional Neural Networks (CNNs)**. The model is trained to classify images into three categories:

- **Potato___Early_blight**
- **Potato___Late_blight**
- **Potato___healthy**

Using **TensorFlow** and **Streamlit**, the project provides an interactive web-based application to predict plant diseases from uploaded images.

## Dataset Structure
The dataset consists of images categorized into training, validation, and testing sets:

```
 dataset/
 ├── Train/
 │   ├── Potato___Early_blight/
 │   ├── Potato___Late_blight/
 │   ├── Potato___healthy/
 │
 ├── Valid/
 │   ├── Potato___Early_blight/
 │   ├── Potato___Late_blight/
 │   ├── Potato___healthy/
 │
 ├── Test/
 │   ├── Potato___Early_blight/
 │   ├── Potato___Late_blight/
 │   ├── Potato___healthy/
```

## Model Architecture
A **Convolutional Neural Network (CNN)** is used for image classification. The model consists of multiple convolutional layers followed by **MaxPooling** layers to extract features. The final layers include **fully connected (dense) layers** and a **softmax activation** function for classification.

### Model Summary:
- **Convolutional layers** with ReLU activation
- **MaxPooling layers** to reduce spatial dimensions
- **Dropout layers** for regularization
- **Fully connected dense layers** for classification
- **Adam optimizer** with a learning rate of 0.0001

## Training Process
The model is trained using the **categorical cross-entropy** loss function and the **Adam optimizer**. Training involves feeding batches of images into the model for **10 epochs**. The performance is monitored using accuracy and loss metrics.

**Final Model Performance:**
- **Validation Accuracy: 92.33%**
- **Validation Loss: 0.18**

The model is saved as:
```
potato.keras
```

## Streamlit Web Application
The **Streamlit app** (`app.py`) provides a user-friendly interface to classify potato leaf diseases.

### Features:
- **Home Page:** Introduction to the system.
- **Disease Recognition:** Upload an image, display it, and classify it using the trained model.
- **Model Predictions:** The CNN model predicts the class of the uploaded image and displays the result.

### How to Run the Streamlit App
1. Clone the repository and navigate to the project directory:
   ```sh
   git clone https://github.com/Sanket77Shanbhag/AICTE-Internship.git
   cd "AICTE-Internship\Potato Leaf Disease Detection"
   ```
2. Install dependencies using the `requirements.txt` file:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
4. Upload an image and view predictions.

## Future Improvements
- **Enhancing model accuracy** with more training data.
- **Implementing transfer learning** using pre-trained models like ResNet.
- **Deploying the model** as a cloud-based API.

## Conclusion
This project demonstrates how **deep learning** can be applied to **agriculture** for disease detection. By deploying this model through a web interface, farmers and researchers can efficiently diagnose plant diseases, improving crop health and sustainability.

