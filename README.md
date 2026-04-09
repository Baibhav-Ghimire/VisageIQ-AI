# 🧠 VisageIQ AI - Face Recognition & Mood Detection System

**Real-time AI-powered Face Recognition and Emotion Analysis System** built with Streamlit.

![Demo](images/live_detection.png)

## ✨ Features

- 📷 **Live Camera Detection** with real-time results
- 👤 **Face Recognition** using deep learning embeddings
- 😊 **Mood / Emotion Detection** (Happy, Sad, Neutral, etc.)
- ➕ **Register New Faces** via camera or image upload
- 👥 **Manage Registered Faces** (View & Delete)
- 🎨 Modern and clean dark-themed UI
- ⚡ Hybrid Detection Pipeline (Fast + Accurate)

## 🧠 AI Technologies Used

- **Face Detection**: Haar Cascade + CNN Fallback
- **Face Recognition**: 128-dimensional Face Embeddings + Similarity Matching
- **Mood Detection**: CNN-based Emotion Classification
- **Real-time Processing**: Optimized pipeline using OpenCV

This project demonstrates a practical **Hybrid AI approach** combining classical computer vision and deep learning.

## 📸 Screenshots

![Live Detection](images/live_detection.png)
![Register Face](images/register.png)
![Manage Faces](images/manage_faces.png)

## 🛠️ Tech Stack

- **Python**
- **Streamlit** (UI)
- **OpenCV**
- **NumPy**
- **Pillow**
- **face_recognition** (dlib)

## ⚡ Installation & Setup

```bash
git clone https://github.com/Baibhav-Ghimire/VisageIQ-AI.git
cd VisageIQ-AI

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate     # For Windows

pip install -r requirements.txt

# Run the app
streamlit run app.py


---------------------------------------------------------------------------------------------------
📁 Project Structure

VisageIQ-AI/
├── app.py                 # Main Streamlit application
├── requirements.txt
├── README.md
├── src/
│   └── detectors/
│       └── chain_master.py   # Core AI engine
├── data/                  # Stores face encodings (auto-created)
├── images/                # Screenshots
└── run.bat

-------------------------------------------------------------------------------------------------

🎯 Use Cases

Smart Attendance System
Office / School Security
Emotion Analysis Tool
AI-based User Interaction

🚀 Future Improvements

Real-time video streaming instead of single frames
Cloud deployment (Streamlit Community Cloud / Hugging Face)
Database integration (SQLite / PostgreSQL)
Attendance reports and analytics
Mobile responsiveness


---------------------------------------------------------------------------------------------------------------
👨‍💻 Author
Baibhav Ghimire
Kathmandu, Nepal
