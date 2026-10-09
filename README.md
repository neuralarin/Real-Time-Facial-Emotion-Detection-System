# Real-Time Facial Emotion Detection System 😊

A deep learning project that detects faces through a webcam and classifies facial expressions into seven emotion categories using a **PyTorch CNN**, **MediaPipe**, and a **Streamlit** web application.

## 🧠 What I Learned

Through this project, I learned how to:

* Work with a facial emotion classification dataset
* Perform image preprocessing, resizing, and normalization
* Build a **CNN (Convolutional Neural Network)** using PyTorch
* Train and evaluate a deep learning image classification model
* Work with PyTorch tensors and neural network layers
* Apply dropout to help reduce overfitting
* Save and load a trained model using `.pth`
* Detect faces in real time using MediaPipe
* Process webcam video frames using OpenCV
* Make real-time predictions using a trained deep learning model
* Reduce prediction fluctuations using temporal smoothing
* Build a simple **Streamlit UI** for real-time emotion detection
* Integrate a trained deep learning model with a webcam-based application

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **OpenCV**
* **MediaPipe**
* **Streamlit**
* **Streamlit-WebRTC**
* **NumPy**
* **Jupyter Notebook**

## 🔄 Project Workflow

**Facial Image Dataset → Image Preprocessing → CNN → Model Training → Evaluation → Model Saving → Face Detection → Emotion Prediction → Streamlit Web Application**

## 🚀 Streamlit Application

The application allows users to:
1. Start their webcam through the Streamlit interface.
2. Detect faces in real time using MediaPipe.
3. Classify facial expressions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
4. Display the predicted emotion and confidence score.
5. Reduce rapid emotion changes using prediction smoothing.

## 📝 Note

**Note:** The Streamlit deployment was done with the help of ChatGPT to explore how my trained model performs in a real-world application.

## 📂 Project Structure

```text
├── model/
│   └── emotion_classifier.pth
├── notebooks/
│   └── Notebook.ipynb
├── app.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 📌 Project Type

**Learning Project** — built to practice the fundamentals of **Deep Learning, CNNs, PyTorch, real-time face detection, computer vision, and Streamlit application development**.
