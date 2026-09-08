# Shoplifting Detection Using 3D CNN

A deep learning computer vision project for **shoplifting detection from video clips** using a **3D Convolutional Neural Network (3D CNN)** built with TensorFlow/Keras.

The model classifies video sequences into two categories:

* **Shop Lifters**
* **Non-Shop Lifters**

Instead of processing individual images, the model learns both **spatial features** from video frames and **temporal patterns** across consecutive frames.

---

## Project Overview

Shoplifting detection is a video classification problem where the model needs to understand what is happening across a sequence of frames.

The project follows this pipeline:

```text
Video Dataset
     ↓
Frame Sampling
     ↓
Frame Resizing & Normalization
     ↓
Train / Validation / Test Split
     ↓
3D CNN
     ↓
Binary Classification
     ↓
Model Evaluation
```

Each video is represented using **16 frames**, with every frame resized to **128 × 128 pixels** and normalized to the range `[0, 1]`.

The resulting input shape for each video is:

```text
(16, 128, 128, 3)
```

---

## Dataset

The dataset contains two classes:

```text
Shop DataSet/
├── non shop lifters/
└── shop lifters/
```

After removing duplicate videos from the non-shoplifting class, the dataset contained:

| Class            |  Videos |
| ---------------- | ------: |
| Non-Shop Lifters |     313 |
| Shop Lifters     |     324 |
| **Total**        | **637** |

The processed dataset has the following shape:

```text
X: (637, 16, 128, 128, 3)
y: (637,)
```

The classes are encoded as:

```python
CLASSES = ['non shop lifters', 'shop lifters']
```

with:

```text
0 → Non-Shop Lifters
1 → Shop Lifters
```

---

## Data Preprocessing

### 1. Frame Sampling

Rather than using every frame in a video, **16 frames are uniformly sampled** throughout each video.

```python
frame_indices = np.linspace(
    0,
    total_frames - 1,
    num_frames,
    dtype=int
)
```

This allows the model to capture information from different points throughout the video while keeping the computational requirements manageable.

### 2. Resizing

Each selected frame is resized to:

```text
128 × 128
```

### 3. Normalization

Pixel values are normalized from `[0, 255]` to `[0, 1]`:

```python
frame = frame / 255.0
```

### 4. Duplicate Handling

Duplicate files were filtered from the non-shoplifting videos based on the dataset's filename convention.

---

## Train / Validation / Test Split

The dataset was divided using `train_test_split`:

```text
Training:   407 videos
Validation: 102 videos
Testing:    128 videos
```

This corresponds approximately to:

```text
64% Training
16% Validation
20% Testing
```

The split was performed using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

followed by another split to create the validation set.

---

## Model Architecture

The project uses a **3D Convolutional Neural Network**.

Unlike a standard `Conv2D`, a `Conv3D` operates across three dimensions, allowing the network to learn patterns across:

* Height
* Width
* Time / Frames

### Architecture

```text
Input
(16, 128, 128, 3)
        │
        ▼
Conv3D
32 filters
3 × 3 × 3
        │
        ▼
MaxPooling3D
2 × 2 × 2
        │
        ▼
Conv3D
64 filters
3 × 3 × 3
        │
        ▼
MaxPooling3D
2 × 2 × 2
        │
        ▼
Flatten
        │
        ▼
Dense
128 neurons
ReLU
        │
        ▼
Dropout
0.5
        │
        ▼
Dense
1 neuron
Sigmoid
        │
        ▼
Binary Prediction
```

### Model Parameters

The network contains approximately:

```text
33.6 Million parameters
```

Specifically:

```text
Total parameters: 33,612,673
Trainable parameters: 33,612,673
```

The majority of the parameters are located in the fully connected layer following the `Flatten` operation.

---

## Training

The model was compiled using:

```python
optimizer = Adam(learning_rate=0.0001)
loss = binary_crossentropy
metric = accuracy
```

Training configuration:

```text
Maximum Epochs: 50
Optimizer: Adam
Learning Rate: 0.0001
Loss: Binary Crossentropy
Batch Size: Default Keras batch size
Early Stopping Patience: 20
```

Early stopping was configured to monitor validation loss and restore the best model weights:

```python
EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)
```

---

## Results

The final evaluation produced the following results:

| Dataset    |   Loss | Accuracy |
| ---------- | -----: | -------: |
| Training   | 0.0088 | **100%** |
| Validation | 0.0268 | **100%** |
| Test       | 0.0191 | **100%** |

The classification report on the test set also showed perfect scores for both classes:

| Class            | Precision | Recall | F1-Score |
| ---------------- | --------: | -----: | -------: |
| Non-Shop Lifters |      1.00 |   1.00 |     1.00 |
| Shop Lifters     |      1.00 |   1.00 |     1.00 |
| **Accuracy**     |           |        | **1.00** |

The test set contained **128 videos**, with 64 samples from each class.

---

## Evaluation

Several evaluation techniques were used:

### Accuracy

Measures the overall percentage of correctly classified videos.

### Precision

Measures how many videos predicted as a particular class actually belong to that class.

### Recall

Measures how many videos belonging to a class were successfully detected.

### F1-Score

Provides a balance between precision and recall.

### Confusion Matrix

A confusion matrix was generated to visualize the classification performance of the model.

```text
                  Predicted
              Non-Shop   Shop
Actual
Non-Shop          ✓        ✗
Shop              ✗        ✓
```

The reported test results produced perfect classification across the two classes.

---

## Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Computer Vision

* OpenCV

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

---

## Project Structure

A recommended repository structure is:

```text
Shoplifting-Detection/
│
├── notebook/
│   └── shoplifting_detection.ipynb
│
├── models/
│   └── shoplifting_model.keras
│
├── README.md
├── requirements.txt
└── .gitignore
```

The original notebook performs dataset loading, video processing, model training, and evaluation in a single workflow.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Shoplifting-Detection.git
cd Shoplifting-Detection
```

Install the required libraries:

```bash
pip install numpy pandas opencv-python matplotlib seaborn scikit-learn tensorflow
```

---

## Usage

1. Download and prepare the shoplifting video dataset.
2. Organize the videos into the two class directories:

```text
Shop DataSet/
├── non shop lifters/
└── shop lifters/
```

3. Update the dataset paths in the notebook:

```python
NON_SHOP_LIFTER_PATH = "path/to/non shop lifters"
SHOP_LIFTER_PATH = "path/to/shop lifters"
```

4. Run the notebook.

The notebook will:

* Load the videos
* Sample 16 frames from each video
* Resize frames to 128 × 128
* Normalize pixel values
* Split the dataset
* Build the 3D CNN
* Train the model
* Evaluate the model
* Generate classification metrics
* Generate a confusion matrix

---

## Key Learning Outcomes

This project demonstrates practical experience with:

* Video preprocessing
* Frame extraction and sampling
* 3D convolutional neural networks
* Temporal feature learning
* Binary video classification
* TensorFlow/Keras model development
* Model training and early stopping
* Classification metrics
* Confusion matrix analysis
* Computer vision workflows

---

## Limitations

Although the reported evaluation results are excellent, the dataset is relatively small, containing **637 videos**. Therefore, additional validation on videos from different environments, stores, camera angles, lighting conditions, and subjects would be valuable before considering the model production-ready.

The current architecture also contains approximately **33.6 million parameters**, making it relatively computationally expensive compared with lighter video-classification architectures.

---

## Future Improvements

Potential improvements include:

* Increasing the size and diversity of the dataset
* Applying data augmentation
* Using a pretrained video model
* Experimenting with CNN + LSTM architectures
* Using transfer learning
* Reducing the number of parameters
* Testing on unseen real-world surveillance footage
* Adding real-time video inference
* Optimizing the model for edge deployment

---

## Conclusion

This project implements a complete **video-based shoplifting detection pipeline** using a 3D CNN.

By processing sequences of frames rather than isolated images, the model can learn both spatial information and temporal patterns within videos. The final experiment achieved **100% accuracy on the training, validation, and test sets**, with perfect precision, recall, and F1-score reported on the test set.

The project provides a practical demonstration of applying deep learning and computer vision techniques to a real-world video classification problem.
