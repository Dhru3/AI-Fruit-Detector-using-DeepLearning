# 🍎 Fruit Detection System using CNN

## 📌 Overview
This project is a Fruit Classification Model that utilizes Deep Learning to identify various types of fruits from images. It leverages a **Convolutional Neural Network (CNN)** built with **TensorFlow** and **Keras** to classify images into multiple fruit categories using the **Fruits-360** dataset.

The model is trained on a dataset of fruit images, which are preprocessed and fed into a CNN architecture to learn distinguishing features. After training, the model can accurately classify images of different fruits based on their visual characteristics.

## 🖥️ Hardware Requirements
To ensure smooth execution, the following hardware specifications are recommended:
- **CPU**: Intel i5/i7 or AMD equivalent (for basic training)
- **GPU**: NVIDIA RTX 2060+ (Recommended for faster training)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space (Dataset + Dependencies)

## 🛠️ Software Requirements
- **Operating System**: Windows 10/11, Linux (Ubuntu recommended), macOS
- **Python**: 3.8+
- **TensorFlow/Keras**: 2.6+
- **Jupyter Notebook** *(optional for running .ipynb files)*

## 📦 Installation
Here are the exact commands needed to set up your environment for running the fruit classification project:

### 1. **Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv fruit_env
source fruit_env/bin/activate  # On macOS/Linux
fruit_env\Scripts\activate  # On Windows
```

### 2. **Install Required Dependencies**
```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn
```

If you're using a **GPU**, install TensorFlow with GPU support:
```bash
pip install tensorflow-gpu
```

### 3. **Install Jupyter Notebook (If Needed)**
```bash
pip install jupyter notebook
```

### 4. **Install Any Missing Dependencies (Based on Errors)**
If you run into missing modules, use:
```bash
pip install <module_name>
```



## 📂 Dataset
The dataset is structured as follows:
```
fruits-360/
│── Training/
│   ├── Apple/
│   ├── Banana/
│   ├── ...
│── Test/
│   ├── Apple/
│   ├── Banana/
│   ├── ...
```
Ensure the dataset is stored in the right file and plug it in the correct places. 

## 📚 Libraries Used
| Library  | Purpose |
|----------|---------|
| `numpy`  | Efficient numerical operations |
| `opencv` | Image processing, webcam access |
| `tensorflow` | Deep learning, neural networks |

## 🏗️ Methodology
The fruit detection system follows these key steps:
1. **Importing Libraries:** TensorFlow for deep learning, OpenCV for image processing.
2. **Loading Pre-trained Model:** A model trained on fruit datasets is used.
3. **Capturing Image Frames:** Webcam captures live video/images for processing.
4. **Preprocessing:** Frames are resized, color-adjusted, and noise is reduced.
5. **Detection & Classification:** The processed frame is analyzed using the model.
6. **Visualization:** The detected fruit is marked with bounding boxes & labels.
7. **Real-time Processing:** The system continuously processes frames in a loop.

## 🦾 Uses and Applications
1. **Agriculture & Farming**
   - Automated **fruit sorting & grading**.
   - **Disease monitoring** in orchards.
   - **Precision harvesting** for better crop yield.

2. **Retail & Consumer Applications**
   - **Smart checkout** systems in grocery stores.
   - **Freshness tracking** for quality control.
   - **Personalized food recommendations**.

3. **Research & Development**
   - **Fruit variety classification**.
   - **Yield estimation** for better farming decisions.
   - **Fruit disease detection**.

## 🚀 Future Improvements
- Implementing **data augmentation** for better generalization.
- Trying **Transfer Learning** (e.g., using MobileNet, ResNet).
- Hyperparameter tuning to optimize the model.
- **Integration with robotics** for **automated harvesting**.
- **Mobile app** for instant fruit detection.
- **More robust model** for extreme lighting conditions.

## 🏆 Acknowledgments
- **Fruits-360 Dataset** by Horea Muresan
- TensorFlow and Keras Documentation

---
Made with ❤️ by **Dhruti Purushotham**

