"""
This script demonstrates the use of Support Vector Machine (SVM) and Neural Network models
for classification on a wheat dataset.

The dataset is loaded from a CSV file, and the features and labels are extracted.
The script then splits the data into training and testing sets for both SVM and Neural Network models.
It trains an SVM classifier with a linear kernel, makes predictions, and evaluates the model.
Next, a Neural Network model is built, trained, and evaluated on the same dataset.

Note: The dataset file 'wheat.csv' should be available in the 'data' directory.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Load the data
data = pd.read_csv('data/wheat.csv')

# Extract features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets for SVM
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the SVM model
svm_classifier.fit(X_train_svm, y_train_svm)

# Make predictions on the test set
y_pred_svm = svm_classifier.predict(X_test_svm)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test_svm, y_pred_svm)
print(f'SVM Test Accuracy: {accuracy_svm}')

# Display SVM confusion matrix
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test_svm, y_pred_svm))

# Split the data into training and testing sets for Neural Network
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode the class labels for Neural Network
encoder = LabelEncoder()
y_train_nn = encoder.fit_transform(y_train_nn)
y_test_nn = encoder.transform(y_test_nn)

# Standardize the features for Neural Network
scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train_nn)
X_test_nn = scaler.transform(X_test_nn)

# Build a neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train_nn.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Neural Network model
model.fit(X_train_nn, y_train_nn, epochs=100, batch_size=32, validation_data=(X_test_nn, y_test_nn), verbose=False)

# Evaluate the Neural Network model on the test set
loss_nn, accuracy_nn = model.evaluate(X_test_nn, y_test_nn)
print(f'Neural Network Test Accuracy: {accuracy_nn}')

# Make predictions on the test set using Neural Network
y_pred_nn_probs = model.predict(X_test_nn)
y_pred_nn = (y_pred_nn_probs > 0.5).astype(int)

# Display Neural Network confusion matrix
print("Neural Network Confusion Matrix:")
print(confusion_matrix(y_test_nn, y_pred_nn))
