# 🚦 Traffic Sign Detection with YOLOv8

![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-orange?logo=roboflow)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-blue?logo=ultralytics)
![Colab](https://img.shields.io/badge/Trained-Google%20Colab-yellow?logo=googlecolab)

A computer vision project for detecting and classifying traffic signs using YOLOv8. The model achieves high performance with **92% mAP50**, **93% precision**, and **88% recall** across 8 traffic sign classes.

## 🚀 Live Demo

Check out the live deployment here: **[Traffic Sign Detector App](https://traffic-sign-detector.streamlit.app/)**

➡️ **Try it out by uploading an image or using your webcam!**

## 📋 Project Overview

This project implements a robust traffic sign detection system that can identify and classify various traffic signs in real-time. The solution leverages state-of-the-art YOLOv8 architecture trained on a carefully curated dataset from Roboflow.

### 🎯 Supported Classes

The model detects the following 8 traffic sign classes:
- `Bump` ⚠️ - Speed bump warning signs
- `No_Parking` 🚫🧊 - No parking zone indicators
- `No_Stopping` 🚫✋ - No stopping restrictions
- `No_U-Turn` 🔃❌ - U-turn prohibition signs
- `Pedestrian` 🚶 - Pedestrian crossing warnings
- `Road_Work` 🚧 - Road work/construction signs
- `Speed_Limit` 🏁 - Speed limit indicators
- `Stop` 🛑 - Stop signs

## 🏗️ Architecture

1. **Data Source**: 📊 Roboflow dataset with annotated traffic signs
2. **Model**: 🤖 YOLOv8 (You Only Look Once version 8)
3. **Training Platform**: ⚡ Google Colab with GPU acceleration
4. **Deployment**: 🌐 Streamlit web application
5. **Performance**: ✅ 92% mAP50, 93% precision, 88% recall

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **mAP50** | **92%** 🏆 |
| **Precision** | **93%** 🎯 |
| **Recall** | **88%** 🔍 |

# Load the trained model
model = YOLO('best.pt')

# Run inference
results = model('your_image.jpg')

# Display results
results[0].show()
📁 Project Structure
text
traffic-sign-detection/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── best.pt               # Trained YOLOv8 model
├── utils/                # Utility functions
│   ├── inference.py      # Inference helpers
│   └── visualization.py  # Visualization tools
├── data/                 # Dataset and configuration
│   └── dataset.yaml      # Dataset configuration file
└── notebooks/            # Jupyter notebooks
    └── training.ipynb    # Training notebook for Colab
🧪 Training Details
The model was trained on Google Colab using the following configuration:

Pretrained weights: YOLOv8l (large version)

Image size: 640x640 pixels

Batch size: 16

Epochs: 100

Optimizer: AdamW

Augmentation: Mosaic, flip, rotation, and color adjustments

🌐 Deployment
The application is deployed on Streamlit Community Cloud, providing:

🔄 Real-time traffic sign detection

📸 Image upload functionality

🌐 Webcam integration for live detection

📊 Results visualization with bounding boxes and confidence scores

🔮 Future Enhancements
📈 Expand to more traffic sign classes

🎥 Implement real-time video processing

📱 Add model quantization for edge deployment

📲 Develop mobile application version

🗺️ Integrate with navigation systems

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Roboflow for providing the annotated dataset

Ultralytics for the YOLOv8 implementation

Google Colab for free GPU resources

Streamlit for easy web app deployment

📧 Contact
For questions or suggestions about this project, please open an issue or contact the maintainer.























