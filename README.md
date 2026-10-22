---
title: Gastric Landmark AI
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# Gastric Landmark AI

A lightweight AI demo for classifying anatomical regions in gastric endoscopy images.

Upload an endoscopic image to obtain a predicted anatomical region and view the model's probability distribution across five classes.

## Features

- Upload endoscopic images in JPG, JPEG, PNG, BMP, and TIFF formats.
- Predict one of five gastric anatomical regions.
- Display the predicted region and its probability.
- Visualize probabilities for all five classes.
- Interactive web interface built with Streamlit and PyTorch.

## Supported Classes

The model classifies images into five anatomical regions:

1. **Cardia**
2. **Fundus**
3. **Greater curvature**
4. **Angulus**
5. **Antrum**

## How to Use

1. Open the application.
2. Upload an endoscopic image.
3. Click **Classify Image**.
4. View the predicted region and class probabilities.

## Model Inference

The application loads a trained PyTorch checkpoint from `best.pth`.

Input images are processed as follows:

1. Convert the image to RGB.
2. Resize it to `224 × 224` pixels.
3. Convert it to a tensor.
4. Normalize using the ImageNet mean and standard deviation.

The model outputs class scores, which are converted into probabilities using softmax.

## Repository Structure

```text
.
├── app.py
├── models.py
├── best.pth
├── requirements.txt
└── README.md
```

| File | Description |
|---|---|
| `app.py` | Streamlit interface and prediction pipeline |
| `models.py` | Model architecture and model creation function |
| `best.pth` | Trained model checkpoint |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Deployment

This repository is configured for Hugging Face Spaces using the Streamlit SDK.

Ensure that `app.py`, `models.py`, `best.pth`, and `requirements.txt` are present in the repository. The checkpoint must be accessible at the location expected by the application.

## Disclaimer

This application is intended for research demonstration and educational purposes only.

Predicted probabilities do not necessarily represent calibrated confidence or clinical accuracy. This tool is not intended for clinical diagnosis or treatment decisions.
