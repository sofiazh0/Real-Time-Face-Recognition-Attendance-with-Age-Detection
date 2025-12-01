# Real-Time Face Recognition Attendance with Age Detection

An automated attendance tracking system that uses machine learning and facial recognition to identify individuals and estimate age from webcam video streams in real-time.

## Description

This project implements an intelligent attendance management system that leverages computer vision and machine learning to automate attendance tracking. The system captures video from a webcam, detects faces using DeepFace, identifies individuals using ensemble machine learning models, estimates their age, and automatically generates timestamped attendance records with saved frame evidence.

## Features

- **Real-time Face Detection**: Uses DeepFace with OpenCV backend for accurate face detection
- **Multi-Model Classification**: Trains and compares 11 different ML classifiers to find the best performing model
- **Age Estimation**: Predicts age using regression models
- **Automated Attendance Logging**: Generates CSV files with timestamps and image references
- **Frame Capture**: Saves annotated frames with bounding boxes and labels
- **Video Demo Generation**: Creates demo videos showing the system in action
- **Model Comparison**: Evaluates multiple algorithms and selects the best performers
- **Support for Multiple Identities**: Trained on multiple individuals (Dua, Jenna, Olivia, Taylor)

## Technologies Used

- **Python 3.x**
- **OpenCV**: Computer vision and image processing
- **DeepFace**: Face detection and analysis
- **scikit-learn**: Machine learning models and utilities
- **XGBoost**: Gradient boosting classifier
- **LightGBM**: Light Gradient Boosting Machine
- **NumPy & Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Joblib**: Model serialization

## Installation

### Prerequisites

Make sure you have Python 3.7+ installed on your system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/sofiazh0/Real-Time-Face-Recognition-Attendance-with-Age-Detection.git
cd face-recognition-attendance-system
```

### Step 2: Install Required Packages

```bash
pip install opencv-python numpy pandas scikit-learn deepface matplotlib joblib xgboost lightgbm
```

Or create a requirements.txt file:

```bash
pip install -r requirements.txt
```

### Step 3: Prepare Training Data

Place your training images in the `Training Data/` folder with the naming convention:
- `name1.png`, `name2.png`, etc.

Update the `data_csv_file.csv` with image paths, identities, and ages.

## Usage

### Training the Models

Run the Jupyter notebook `Main.ipynb` or execute the cells sequentially:

1. **Load and prepare the dataset**
2. **Train multiple classifiers** (SVM, Random Forest, XGBoost, etc.)
3. **Train regressors** for age estimation
4. **Save the best models**

### Running the Attendance System

The system will:
1. Activate your webcam
2. Detect faces in real-time
3. Identify individuals
4. Estimate age
5. Log attendance with timestamps
6. Save annotated frames

```python
# The video processing runs automatically for 100 frames
# Adjust the range in the notebook for longer/shorter sessions
```

### Output Files

- `attendance_YYYY-MM-DD.csv`: Attendance log with names, times, and image paths
- `frames/YYYY-MM-DD/`: Directory containing saved frames
- `demo_video.mp4`: Video demonstration of the system
- `best_classifier_model.pkl`: Trained classification model
- `best_regressor_model.pkl`: Trained regression model
- `label_encoder.pkl`: Label encoder for identity mapping

## 📊 Model Performance

### Classification Models Tested

| Model | Accuracy |
|-------|----------|
| Random Forest | 65.4% |
| XGBoost | 65.4% |
| KNN | 55.8% |
| Naive Bayes | 51.9% |
| SVC (Linear) | 50.0% |
| Gradient Boosting | 50.0% |
| Decision Tree | 46.2% |
| Logistic Regression | 46.2% |
| AdaBoost | 44.2% |
| MLP Classifier | 38.5% |
| LightGBM | Varies |

**Best Classifier**: Random Forest / XGBoost (65.4% accuracy)

### Regression Models (Age Estimation)

| Model | Mean Squared Error |
|-------|-------------------|
| Random Forest Regressor | 1.18 |
| SVR | 1.22 |
| Linear Regression | 2.92 |

**Best Regressor**: Random Forest Regressor (MSE: 1.18)

## 🏗️ Project Structure

```
face-recognition-attendance-system/
├── Main.ipynb          # Main Jupyter notebook
├── data_csv_file.csv           # Dataset with image paths and labels
├── best_classifier_model.pkl   # Saved classifier model
├── best_regressor_model.pkl    # Saved regressor model
├── label_encoder.pkl           # Label encoder
├── Training Data/              # Training images
│   ├── dua1.png
│   ├── jenna1.jpeg
│   ├── olivia1.png
│   ├── taylor1.png
│   └── ...
├── frames/                     # Captured frames during attendance
│   └── YYYY-MM-DD/
│       └── HHMMSS_Name.png
├── attendance_YYYY-MM-DD.csv   # Attendance records
└── README.md                   # This file
```

## How It Works

1. **Data Preparation**: Training images are loaded, resized to 32x32 pixels, converted to grayscale, and flattened
2. **Model Training**: Multiple ML models are trained on the prepared dataset
3. **Model Selection**: The best performing classifier and regressor are selected based on accuracy and MSE
4. **Face Detection**: DeepFace detects faces in video frames using OpenCV backend
5. **Prediction**: Detected faces are preprocessed and fed to the trained models
6. **Attendance Logging**: Identified individuals are logged with timestamps
7. **Frame Saving**: Annotated frames with bounding boxes and labels are saved

## Key Algorithms

### Classification
- **Support Vector Machine (SVM)**: Linear kernel for face classification
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting with tree-based learners
- **K-Nearest Neighbors**: Distance-based classification
- **Neural Networks**: Multi-layer perceptron classifier

### Regression
- **Random Forest Regressor**: Ensemble method for age prediction
- **Support Vector Regression**: Non-linear age estimation
- **Linear Regression**: Baseline regression model

## Future Improvements

- [ ] Add support for more training data to improve accuracy
- [ ] Implement deep learning models (CNNs) for better face recognition
- [ ] Add face verification for enhanced security
- [ ] Implement database integration for attendance records
- [ ] Add web interface for easy access
- [ ] Support for multiple camera streams
- [ ] Add emotion detection
- [ ] Implement attendance analytics and reporting
- [ ] Add real-time notification system
- [ ] Support for mask detection

## Notes

- The system processes 100 frames by default; adjust as needed
- Webcam must be available and accessible
- Training data quality significantly impacts model performance
- TensorFlow warnings are suppressed for cleaner output

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) for face detection framework
- scikit-learn for machine learning tools
- OpenCV for computer vision capabilities

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This system is designed for educational purposes. For production use, consider additional security measures and privacy compliance (GDPR, CCPA, etc.).

