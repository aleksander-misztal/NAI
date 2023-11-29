import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import tree, svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd


class CarData:
    def __init__(self, file_path):
        """
        Initialize the CarData class.

        Parameters:
        - file_path (str): The file path to the CSV containing car data.
        """
        self.file_path = file_path
        self.load_and_clean_data()

    def load_and_clean_data(self):
        """
        Load and clean the car data from the CSV file.
        """
        self.cars_data = pd.read_csv(self.file_path)
        for col in self.cars_data.columns:
            self.cars_data.rename(columns={col: col.strip()}, inplace=True)

        self.cars_data['weightlbs'] = pd.to_numeric(
            self.cars_data['weightlbs'], errors='coerce')
        self.cars_data['cubicinches'] = pd.to_numeric(
            self.cars_data['cubicinches'], errors='coerce')

        self.cars_data['cubicinches'] = self.cars_data['cubicinches'].fillna(
            value=self.cars_data['cubicinches'].mode()[0])
        self.cars_data['weightlbs'] = self.cars_data['weightlbs'].fillna(
            value=self.cars_data['weightlbs'].mean())

    def split_data(self):
        """
        Split the data into training and testing sets.

        Returns:
        - tuple: X_train, X_test, Y_train, Y_test
        """
        X = self.cars_data.iloc[:, :7]
        Y = self.cars_data.iloc[:, -1]
        return train_test_split(X, Y, test_size=0.25, random_state=20)


class DecisionTreeClassifierClass(CarData):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.model = None
        self.best_params = None

    def split_data(self):
        """
        Override the split_data method to include stratification for classification tasks.
        """
        X = self.cars_data.iloc[:, :7]
        Y = self.cars_data.iloc[:, -1]
        return train_test_split(X, Y, test_size=0.25, random_state=20, stratify=Y)

    def tune_hyperparameters(self):
        """
        Tune hyperparameters for Decision Tree using GridSearchCV.
        """
        print("Tuning hyperparameters for Decision Tree...")
        X_train, _, Y_train, _ = self.split_data()

        param_grid = {'max_depth': range(1, 10)}
        grid_search = GridSearchCV(
            tree.DecisionTreeClassifier(random_state=0), param_grid, scoring='accuracy', cv=5)
        grid_search.fit(X_train, Y_train)
        self.best_params = grid_search.best_params_
        print("Best hyperparameters:", self.best_params)

    def train(self):
        """
        Train the Decision Tree model.
        """
        print("Training Decision Tree classifier...")
        X_train, x_test, Y_train, y_test = self.split_data()
        self.model = tree.DecisionTreeClassifier(
            max_depth=self.best_params['max_depth'], random_state=0)
        self.model.fit(X_train, Y_train)

    def evaluate(self):
        """
        Evaluate the Decision Tree model on training and testing sets.
        """
        print("Evaluating the Decision Tree model on training and testing sets...")
        X_train, x_test, Y_train, y_test = self.split_data()
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(x_test)

        train_accuracy = accuracy_score(train_pred, Y_train)
        test_accuracy = accuracy_score(test_pred, y_test)

        print(f"Training Accuracy: {train_accuracy}")
        print(f"Testing Accuracy: {test_accuracy}")
        # Confusion Matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, test_pred)
        print(cm)

    def train_and_evaluate(self):
        """
        Train and evaluate the Decision Tree model.
        """
        print("Starting the main function...")
        print("Creating Decision Tree classifier...")
        self.tune_hyperparameters()
        self.train()
        self.evaluate()


class SVMClassifierClass(CarData):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.model = None
        self.best_params = None

    def tune_hyperparameters(self):
        """
        Tune hyperparameters for SVM using GridSearchCV.
        """
        X_train, _, Y_train, _ = self.split_data()

        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [
            1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
        grid_search = GridSearchCV(
            svm.SVC(), param_grid, scoring='accuracy', cv=5)
        grid_search.fit(X_train, Y_train)
        self.best_params = grid_search.best_params_

    def train(self):
        """
        Train the SVM model.
        """
        X_train, x_test, Y_train, y_test = self.split_data()
        self.model = svm.SVC(**self.best_params)
        self.model.fit(X_train, Y_train)

    def evaluate(self):
        """
        Evaluate the SVM model on training and testing sets.

        Returns:
        - tuple: train_accuracy, test_accuracy
        """
        X_train, x_test, Y_train, y_test = self.split_data()
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(x_test)

        train_accuracy = accuracy_score(train_pred, Y_train)
        test_accuracy = accuracy_score(test_pred, y_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test, test_pred)

        return train_accuracy, test_accuracy

    def train_and_evaluate(self):
        """
        Train and evaluate the SVM model.
        """
        self.tune_hyperparameters()
        self.train()
        self.evaluate()


def main():
    """
    Main function to create and train either SVM or Decision Tree classifier.
    """
    use_svm = False  # Set to True for SVM, False for Decision Tree

    if use_svm:
        cars_model = SVMClassifierClass('cars/cars.csv')
    else:
        cars_model = DecisionTreeClassifierClass('cars/cars.csv')

    cars_model.train_and_evaluate()


if __name__ == "__main__":
    main()
