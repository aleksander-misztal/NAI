"""
Sentiment Analysis Model

This script demonstrates sentiment analysis using a Long Short-Term Memory (LSTM) neural network.
The model is trained on a dataset of reviews and predicts whether a given review is positive or negative.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load data from CSV file
data = pd.read_csv('data/rev.csv')

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare data for the model
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['recenzja'])
X_train = tokenizer.texts_to_sequences(train_data['recenzja'])
X_test = tokenizer.texts_to_sequences(test_data['recenzja'])

X_train = pad_sequences(X_train)
X_test = pad_sequences(X_test)

# Prepare labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['etykieta'])
y_test = label_encoder.transform(test_data['etykieta'])

# Build the neural network model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=False)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# Sample classification for a positive review
sample_review_positive = ["To jest naprawdę dobra książka. Polecam!"]
sample_sequence_positive = tokenizer.texts_to_sequences(sample_review_positive)
sample_padded_positive = pad_sequences(sample_sequence_positive, maxlen=X_train.shape[1])
predicted_prob_positive = model.predict(sample_padded_positive)
predicted_label_positive = 1 if predicted_prob_positive[0] > 0.5 else 0
print(f'Predicted label for positive review: {predicted_label_positive}')

# Sample classification for a negative review
sample_review_negative = ["Niestety, film okazał się być dużym rozczarowaniem. Słaba gra aktorska i przewidywalna historia."]
sample_sequence_negative = tokenizer.texts_to_sequences(sample_review_negative)
sample_padded_negative = pad_sequences(sample_sequence_negative, maxlen=X_train.shape[1])
predicted_prob_negative = model.predict(sample_padded_negative)
predicted_label_negative = 1 if predicted_prob_negative[0] > 0.5 else 0
print(f'Predicted label for negative review: {predicted_label_negative}')

# Another sample classification for a negative review
sample_review_negative = ["film byl fatalny, najgorszy jaki widzialem, nie polecam"]
sample_sequence_negative = tokenizer.texts_to_sequences(sample_review_negative)
sample_padded_negative = pad_sequences(sample_sequence_negative, maxlen=X_train.shape[1])
predicted_prob_negative = model.predict(sample_padded_negative)
predicted_label_negative = 1 if predicted_prob_negative[0] > 0.5 else 0
print(f'Predicted label for negative review: {predicted_label_negative}')
