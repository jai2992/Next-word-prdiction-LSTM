# 🌟 Next Word Prediction Using LSTM 🌟

Welcome to the Next Word Prediction app, built using TensorFlow and Streamlit! This app demonstrates two different implementations of an LSTM-based model for predicting the next word in a sentence.

## 🚀 Getting Started

### Prerequisites

Make sure you have the following packages installed:

- TensorFlow
- Streamlit
- Numpy
- Regex
- Matplotlib (for the first app)

You can install the necessary packages using:

```bash
pip install tensorflow streamlit numpy regex matplotlib
```

## First App: Training the Model On-the-Fly

This app allows you to upload a text file and train an LSTM model on-the-fly.

- Upload your text file: Choose a .txt file containing the text data.
- Train the model: The model will be trained and validated on the uploaded text data.
- Predict the next words: Enter some seed text and specify the number of words to predict. The app will generate the next words based on your input.

*screenshots*

![image](https://github.com/jai2992/Next-word-prdiction-LSTM/assets/136327019/55f07abd-6f48-45ab-bf71-374d67985054)

![image](https://github.com/jai2992/Next-word-prdiction-LSTM/assets/136327019/2f57726c-efb4-4687-9e2e-3fd59acaa33f)

![image](https://github.com/jai2992/Next-word-prdiction-LSTM/assets/136327019/01bbc8f2-2ad0-4fac-8108-1baacd6ca12e)


## Second App: Using a Pre-trained Model

This app loads a pre-trained LSTM model to predict the next words in a sentence.

- Load the model and tokenizer: The pre-trained model (model.h5) and tokenizer are loaded.
- Predict the next words: Enter some seed text and specify the number of words to predict. The app will generate the next words based on your input.

*screenshots*

![image](https://github.com/jai2992/Next-word-prdiction-LSTM/assets/136327019/59b980b0-e5bd-4996-8a46-0c249969ec57)

![image](https://github.com/jai2992/Next-word-prdiction-LSTM/assets/136327019/d29a1486-da6d-4137-9336-a9065664d8d0)

## 📂 File Structure


├── first_app.py # Code for the first app (training on-the-fly)

├── second_app.py # Code for the second app (using pre-trained model)

├── model.h5 # Pre-trained model file for the second app

├── data.txt # Sample text data file for the second app

└── README.md  # This README file

## 🎨 Features

Interactive UI: Simple and interactive user interface using Streamlit.
On-the-fly Training: Train an LSTM model on uploaded text data in the first app.
Pre-trained Model: Use a pre-trained model for next word prediction in the second app.
Visualization: Visualize the training process and performance metrics (in the first app).


