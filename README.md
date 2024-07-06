# Game_of_deep_learning_ships_classification

Ship Classification Model: README
Introduction
This README provides an overview of the ship classification model, which is designed to classify images of ships into five categories: Cargo, Military, Carrier, Cruise, and Tankers. The model utilizes a Convolutional Neural Network (CNN) with dropout layers to prevent overfitting and improve generalization.

Dependencies
Ensure you have the following dependencies installed:

TensorFlow
NumPy
Matplotlib
Pandas
You can install the required packages using pip

Data Preparation
The dataset is divided into training and test sets. The training set includes images and their corresponding labels, while the test set contains only the image paths.

Loading and Preprocessing Data
The images are loaded and preprocessed to fit the input shape of the model (128x128x3). The preprocessing steps include rescaling the pixel values to the range [0, 1].


Model Architecture
The model is a Convolutional Neural Network (CNN) with the following layers:

Conv2D: 32 filters, kernel size (3, 3), ReLU activation
MaxPooling2D: Pool size (2, 2)
Dropout: 0.25
Conv2D: 64 filters, kernel size (3, 3), ReLU activation
MaxPooling2D: Pool size (2, 2)
Dropout: 0.25
Conv2D: 128 filters, kernel size (3, 3), ReLU activation
MaxPooling2D: Pool size (2, 2)
Dropout: 0.25
Flatten
Dense: 512 units, ReLU activation
Dropout: 0.5
Dense: 5 units, softmax activation

Model Compilation
The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.

Training
The model is trained using the training set with a validation split for monitoring performance on unseen data. The training process includes:

EarlyStopping: Stop training when the validation loss does not improve for 10 consecutive epochs.
ModelCheckpoint: Save the best model based on validation loss.


Evaluation
The model's performance is evaluated using a confusion matrix and accuracy metrics. The results are visualized through loss and accuracy plots over the training epochs.


Predictions
The model is used to make predictions on the test images. The predicted labels are compared with the actual labels to generate a confusion matrix.

Results
The training and validation accuracy and loss are plotted to visualize the model's performance over the epochs. The confusion matrix provides insights into the model's classification accuracy for each category.

Training and Validation Loss

Training and Validation Accuracy

Conclusion
The ship classification model successfully classifies images of ships into five categories with good accuracy. The inclusion of dropout layers helps in preventing overfitting and improving generalization. Further improvements can be made by fine-tuning the model and experimenting with different architectures.

Contact
For any questions or issues, please contact Ibrahim Naeem at mibra7528@gmail.com.

