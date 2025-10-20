# Emotion Classification using FER-2013 Dataset

A deep learning project that implements a Convolutional Neural Network (CNN) for facial emotion recognition using the FER-2013 dataset. The model can classify facial expressions into 7 different emotion categories.

## 🎯 Project Overview

This project demonstrates a complete machine learning pipeline for emotion classification, including:
- Data preprocessing and cleaning
- Duplicate detection and removal
- Data augmentation for class balancing
- CNN model training and evaluation
- Model saving and loading functionality

## 📊 Dataset Information

**FER-2013 (Facial Expression Recognition 2013)**
- **Classes**: 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise)
- **Image Format**: 48x48 grayscale images
- **Total Images**: ~35,000 training images, ~7,000 test images
- **Class Distribution**: Imbalanced (happy: ~7,000, disgust: ~400)

## 🏗️ Model Architecture

The CNN model consists of:
- **3 Convolutional Layers**: 32, 64, 128 filters respectively
- **Activation**: ReLU activation function
- **Pooling**: Max pooling after each convolutional layer
- **Fully Connected Layers**: 512 hidden units + 7 output classes
- **Regularization**: Dropout (0.5) to prevent overfitting
- **Total Parameters**: ~1.2M parameters

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, for faster training)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd emotion-classification-fer2013
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the FER-2013 dataset**
   - Download from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
   - Extract the dataset to the `./data/` directory
   - Ensure the structure looks like:
     ```
     data/
     ├── train/
     │   ├── angry/
     │   ├── disgust/
     │   ├── fear/
     │   ├── happy/
     │   ├── neutral/
     │   ├── sad/
     │   └── surprise/
     └── test/
         ├── angry/
         ├── disgust/
         ├── fear/
         ├── happy/
         ├── neutral/
         ├── sad/
         └── surprise/
     ```

### Usage

1. **Run the complete pipeline**
   ```bash
   jupyter notebook emotion_classification_FER_2013.ipynb
   ```
   Then run all cells in order.

2. **Train the model**
   - The notebook will automatically:
     - Load and preprocess the data
     - Remove duplicates and corrupted images
     - Augment the minority class (disgust)
     - Train the CNN model
     - Evaluate performance

3. **Make predictions on new images**
   ```python
   # Load a trained model
   model = load_model('emotion_model.pth')
   
   # Predict emotion from an image
   predict_emotion(model, 'path/to/image.jpg', class_names)
   ```

## 📈 Performance Results

The model achieves the following performance on the test set:
- **Accuracy**: ~57% (typical for FER-2013 dataset)
- **F1-Score**: ~57% (weighted average)

### Confusion Matrix
The model shows varying performance across emotion classes:
- **Best**: Happy, Neutral emotions
- **Challenging**: Disgust, Fear emotions (due to class imbalance)

## 🔧 Configuration

Key parameters can be modified in the `CONFIG` dictionary:

```python
CONFIG = {
    'dataset_name': 'fer2013',
    'data_dir': './data',
    'batch_size': 64,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'image_size': 48
}
```

## 📁 Project Structure

```
emotion-classification-fer2013/
├── emotion_classification_FER_2013.ipynb  # Main notebook
├── requirements.txt                      # Python dependencies
├── README.md                            # This file
├── data/                                # Dataset directory
│   ├── train/                          # Training images
│   └── test/                           # Test images
└── emotion_model.pth                    # Saved model (after training)
```

## 🛠️ Data Preprocessing Pipeline

1. **Duplicate Detection**: Uses perceptual hashing to identify and remove duplicate images
2. **Corrupted Image Removal**: Filters out completely black or corrupted images
3. **Data Augmentation**: Generates synthetic images for the minority class (disgust) using:
   - Random rotation (±20°)
   - Random resizing and cropping
   - Random horizontal flipping
   - Color jittering
4. **Normalization**: Standardizes pixel values to [-1, 1] range

## 🎨 Visualization Features

The notebook includes several visualization components:
- Sample images from each emotion class
- Training loss and accuracy curves
- Confusion matrix heatmap
- Class distribution charts

## 🔍 Model Evaluation

The evaluation includes:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1-score across all classes
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Detailed class-wise performance analysis

## 🚨 Common Issues and Solutions

### Issue: "Dataset not found"
**Solution**: Ensure the FER-2013 dataset is downloaded and extracted to `./data/` directory with correct folder structure.

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in the CONFIG dictionary or use CPU training.

### Issue: "Low accuracy"
**Solution**: This is expected for FER-2013 dataset. Consider:
- Increasing training epochs
- Adjusting learning rate
- Using more sophisticated architectures
- Implementing ensemble methods

## 📚 Dependencies

- **torch** (>=1.9.0): PyTorch deep learning framework
- **torchvision** (>=0.10.0): Computer vision utilities
- **opencv-python** (>=4.5.0): Image processing
- **Pillow** (>=8.0.0): Image manipulation
- **matplotlib** (>=3.3.0): Plotting and visualization
- **seaborn** (>=0.11.0): Statistical data visualization
- **scikit-learn** (>=1.0.0): Machine learning utilities
- **numpy** (>=1.21.0): Numerical computing
- **imagehash** (>=4.3.0): Perceptual image hashing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.


## 🙏 Acknowledgments

- **FER-2013 Dataset**: Created by Pierre-Luc Carrier and Aaron Courville
- **PyTorch Team**: For the excellent deep learning framework
- **Kaggle Community**: For hosting the dataset and providing inspiration

## 👤 Author

**Mohamed Galal**  
📍 Cairo University, B.Sc. in Computer Science  
💼 AI Engineer | Python | Computer Vision | NLP  
🔗 [GitHub](https://github.com/galall10) • [LinkedIn](https://linkedin.com/in/mohamedgalal10)
