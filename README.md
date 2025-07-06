# 🌾 Rice Leaf Disease Detection using InceptionV3

A deep learning-based web application for detecting and classifying rice leaf diseases using the InceptionV3 architecture. This project helps farmers and agricultural experts identify common rice diseases early to prevent crop damage and improve yield.

## 🎯 Project Overview

This project implements a computer vision solution for rice leaf disease detection using transfer learning with the InceptionV3 model. The system can classify rice leaves into four categories:
- **Bacterial Blight** - A serious bacterial disease affecting rice plants
- **Blast** - A fungal disease that can cause significant yield loss
- **Brown Spot** - A fungal disease affecting rice leaves and grains
- **Tungro** - A viral disease transmitted by green leafhoppers

## 🚀 Features

- **Real-time Disease Detection**: Upload rice leaf images and get instant disease classification
- **High Accuracy**: Trained on a comprehensive dataset with data augmentation
- **User-friendly Interface**: Streamlit-based web application with intuitive UI
- **Confidence Scoring**: Provides prediction confidence percentages
- **Multiple Disease Classes**: Detects four major rice diseases
- **Transfer Learning**: Utilizes pre-trained InceptionV3 for better performance

## 📁 Project Structure

```
FY_Project/
├── app.py                                    # Streamlit web application
├── Advanced_inceptioV3_sgd.py               # Model training script
├── requirements.txt                          # Python dependencies
├── README.md                                # Project documentation
└── split_dataset/                           # Dataset directory (not included in repo)
    ├── train/                               # Training images
    ├── validation/                          # Validation images
    └── test/                                # Test images
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd FY_Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the trained model**
   - Due to file size limitations, the trained model (`Model_inceptionv3_SGD_Epochs_30_(TRYSplit_dataset).keras`) is not included in the repository
   - Please contact me to obtain the model file or you can train too 🤷
   - Place the model file in the project root directory

## 🚀 Usage

### Running the Web Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - Upload a rice leaf image
   - View the disease classification results

### Using the Application

1. **Upload Image**: Click "Choose a rice leaf image..." and select a JPG, JPEG, or PNG file
2. **View Results**: The application will display:
   - The uploaded image
   - Predicted disease class
   - Confidence percentage

## 🧠 Model Architecture

### InceptionV3 Transfer Learning

The model uses a pre-trained InceptionV3 architecture with the following modifications:

- **Base Model**: InceptionV3 pre-trained on ImageNet
- **Custom Layers**:
  - Flatten layer
  - Dense layer (512 units, ReLU activation)
  - Batch Normalization
  - Dropout (0.5)
  - Dense layer (256 units, ReLU activation)
  - Batch Normalization
  - Dropout (0.5)
  - Output layer (4 units, Softmax activation)

### Training Strategy

1. **Initial Training**: Freeze base model layers, train custom layers
2. **Fine-tuning**: Unfreeze last 50 layers of base model, retrain with lower learning rate
3. **Data Augmentation**: Rotation, width/height shift, zoom, brightness adjustment, horizontal flip
4. **Class Balancing**: Computed class weights to handle imbalanced dataset

## 📊 Dataset

The model was trained on a comprehensive dataset of rice leaf images:

- **Training Set**: 3,124 images (781 per class)
- **Validation Set**: 1,036 images (259 per class)
- **Test Set**: 1,036 images (259 per class)
- **Total Images**: 5,196 images across 4 disease classes

### Dataset Classes
- Bacterialblight: 1,036 images
- Blast: 1,036 images
- Brownspot: 1,036 images
- Tungro: 1,036 images

## 🔧 Training the Model

To retrain the model with your own dataset:

1. **Prepare your dataset** in the following structure:
   ```
   split_dataset/
   ├── train/
   │   ├── Bacterialblight/
   │   ├── Blast/
   │   ├── Brownspot/
   │   └── Tungro/
   ├── validation/
   │   ├── Bacterialblight/
   │   ├── Blast/
   │   ├── Brownspot/
   │   └── Tungro/
   └── test/
       ├── Bacterialblight/
       ├── Blast/
       ├── Brownspot/
       └── Tungro/
   ```

2. **Run the training script**:
   ```bash
   python Advanced_inceptioV3_sgd.py
   ```

3. **Training parameters**:
   - Epochs: 30 (initial) + 30 (fine-tuning)
   - Batch size: 16
   - Learning rate: 0.001 (initial), 0.0001 (fine-tuning)
   - Optimizer: SGD with momentum (0.9)
   - Image size: 224x224 pixels

## 📈 Performance Metrics

The model achieves the following performance:
- **Validation Accuracy**: [To be added after training]
- **Test Accuracy**: [To be added after training]
- **Confusion Matrix**: Available in training output
- **Classification Report**: Detailed precision, recall, and F1-score per class

## 🛡️ Error Handling

The application includes:
- File format validation (JPG, JPEG, PNG only)
- Image preprocessing error handling
- Model loading error handling
- User-friendly error messages

## 🔮 Future Enhancements

- [ ] Add support for more rice diseases
- [ ] Implement batch processing for multiple images
- [ ] Add disease severity assessment
- [ ] Integrate with mobile applications
- [ ] Add treatment recommendations
- [ ] Implement real-time camera detection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Kaushik Karmakar**
- Final Year Project
- Computer Science/Engineering

## 🙏 Acknowledgments

- Rice disease dataset contributors
- TensorFlow and Keras development teams
- Streamlit for the web framework
- Agricultural experts for domain knowledge

## 📞 Contact

For questions, issues, or collaboration opportunities:
- Email: [karmakark1267@gmail.com]
- GitHub: [https://github.com/Kaushik2105]

## ⚠️ Important Notes

- The trained model file (`Model_inceptionv3_SGD_Epochs_30_(TRYSplit_dataset).keras`) is not included due to GitHub file size limitations (>300MB)
- The dataset (`split_dataset/`) is also excluded for the same reason
- Please contact the maintainer to obtain these files if needed for reproduction

---

**Disclaimer**: This tool is designed for educational and research purposes. For commercial agricultural applications, please consult with agricultural experts and validate results in your specific context. 