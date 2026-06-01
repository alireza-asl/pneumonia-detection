#Chest X-Ray Pneumonia Prediction

This project explores how Convolutional Neural Networks (CNNs) can be used to classify chest X-ray images as either *NORMAL* or *PNEUMONIA*.

I built this project as a hands-on learning experience to better understand image preprocessing, CNN architecture, model training, evaluation, and image prediction using TensorFlow and Keras.


#project structure

```text
chest-xray-pneumonia-prediction
│
├── dataset
│   ├── train
│   │   ├── NORMAL
│   │   └── PNEUMONIA
│   ├── validation
│   │   ├── NORMAL
│   │   └── PNEUMONIA
│   └── test
│       ├── NORMAL
│       └── PNEUMONIA
├── model
├── images
├── main.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

The repository includes '.gitkeep' files to preserve the folder structure on GitHub while keeping datasets and trained models out of the repository.



#dataset

Before training, place the chest X-ray images into the appropriate folders:


dataset/
├── train/
├── validation/
└── test/


The model learns two classes:

* NORMAL
* PNEUMONIA

Folder names are used automatically as labels by TensorFlow.



#installation

Clone the repository and install the required packages:


pip install -r requirements.txt


#training

To train the model:


python main.py


The training script will:

 Load images from the dataset folders
 Preprocess and resize images
 Train the CNN model
 Validate performance after each epoch
 Evaluate the final model on the test dataset
 Display training graphs
 Save the trained model

The trained model is stored in:

model/pneumonia_classifier.keras


#model Architecture

The model uses a simple CNN architecture consisting of:

 Rescaling layer
 Convolution layers
 MaxPooling layers
 Dropout layers
 Dense layers
 Sigmoid output layer

The goal was to keep the architecture easy to understand while still demonstrating the core concepts of image classification.


#evaluation

After training is complete, the model is evaluated using the test dataset.

example output:


Final Test Accuracy: 86.5%
Final Test Loss: 0.3412


Results will be different depending on dataset size, image quality, and training settings.


#prediction

Once the model has been trained, predictions can be made on individual X-ray images:

python predict.py path/to/image.jpeg

example output:


#Prediction: PNEUMONIA (94.2%)
or
#Prediction: NORMAL (88.7%)

The prediction script loads the saved model, processes the input image, and returns the predicted class along with a confidence score.


#training experiments

During development, different epoch values were tested to observe their effect on model performance.

Increasing the number of epochs generally improved training accuracy, but it also increased the risk of overfitting. In several experiments, the model continued improving on the training set while showing little or no improvement on unseen data.

Because this project focuses on learning and experimentation rather than production-level medical classification, a relatively small number of epochs was chosen as a reasonable balance between training performance and generalization.

The impact of different epoch values can be observed through the generated accuracy and loss graphs.



#what I learned

Building this project helped me understand the complete workflow of an image classification task:

 Organizing image datasets
 Preprocessing images
 Building CNN models with TensorFlow
 Training and evaluating neural networks 
 Detecting signs of overfitting
 Saving and loading trained models
 Making predictions on new images

Note: This project was created for learning purposes and should not be used for real-world medical diagnosis or healthcare decisions.
