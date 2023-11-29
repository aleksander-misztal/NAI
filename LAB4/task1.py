import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.simplefilter("ignore")


class GrainClassifier:
    """
    GrainClassifier class for processing and preparing data.
    """

    def __init__(self, file_path):
        """
        Constructor to initialize the GrainClassifier.

        Parameters:
        - file_path (str): Path to the CSV file containing data.
        """
        self.df = pd.read_csv(file_path)
        self.df.columns = self.df.columns.str.strip()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_data(self):
        """
        Preprocess the data by encoding 'Class' column, handling missing values, splitting, and scaling.
        """
        print("Preprocessing data...")
        self.handle_missing_values()
        self.label_encode_class()
        self.split_data()
        self.scale_data()

    def handle_missing_values(self):
        """
        Handle missing values in the dataset.
        """
        print("Handling missing values...")
        # You can replace missing values with a strategy suitable for your data
        self.df.fillna(self.df.mean(), inplace=True)

    def label_encode_class(self):
        """
        Label encode the 'Class' column.
        """
        print("Label encoding 'Class' column...")
        self.df['Class'] = self.label_encoder.fit_transform(self.df['Class'])

    def split_data(self):
        """
        Split the data into training and testing sets.
        """
        print("Splitting data into training and testing sets...")
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def scale_data(self):
        """
        Scale the features using StandardScaler.
        """
        print("Scaling features using StandardScaler...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)


class SVMClassifierClass(GrainClassifier):
    """
    SVMClassifierClass extends GrainClassifier for SVM model.
    """

    def __init__(self, file_path):
        """
        Constructor to initialize the SVMClassifierClass.

        Parameters:
        - file_path (str): Path to the CSV file containing data.
        """
        super().__init__(file_path)
        self.model = SVC(random_state=42)
        self.best_params = None

    def tune_hyperparameters(self):
        """
        Tune hyperparameters using GridSearchCV.
        """
        print("Tuning hyperparameters for SVM using GridSearchCV...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1, 1],
        }
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)
        self.best_params = grid_search.best_params_

    def train_model(self):
        """
        Train the SVM model using the best hyperparameters.
        """
        print("Training SVM model using the best hyperparameters...")
        self.model = SVC(**self.best_params)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate and print the performance of the SVM model.
        """
        print("Evaluating SVM model...")
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred)

        print("SVM Model:")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", class_report)


class DecisionTreeClassifierClass(GrainClassifier):
    """
    DecisionTreeClassifierClass extends GrainClassifier for Decision Tree model.
    """

    def __init__(self, file_path):
        """
        Constructor to initialize the DecisionTreeClassifierClass.

        Parameters:
        - file_path (str): Path to the CSV file containing data.
        """
        super().__init__(file_path)
        self.model = DecisionTreeClassifier()
        self.best_params = None

    def tune_hyperparameters(self):
        """
        Tune hyperparameters using GridSearchCV.
        """
        print("Tuning hyperparameters for Decision Tree using GridSearchCV...")
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)
        self.best_params = grid_search.best_params_

    def train_model(self):
        """
        Train the Decision Tree model using the best hyperparameters.
        """
        print("Training Decision Tree model using the best hyperparameters...")
        self.model = DecisionTreeClassifier(**self.best_params)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate and print the performance of the Decision Tree model.
        """
        print("Evaluating Decision Tree model...")
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred)

        print("Decision Tree Model:")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", class_report)


def main():
    """
    Main function to demonstrate the workflow of the classifiers.
    """
    print("Starting the main function...")
    use_svm = False  # Set to True for SVM, False for Decision Tree

    if use_svm:
        print("Creating SVM classifier...")
        grain_model = SVMClassifierClass('data/wheat.csv')
    else:
        print("Creating Decision Tree classifier...")
        grain_model = DecisionTreeClassifierClass('data/wheat.csv')

    print("Preprocessing data...")
    grain_model.preprocess_data()
    print("Tuning hyperparameters...")
    grain_model.tune_hyperparameters()
    print("Training model...")
    grain_model.train_model()
    print("Evaluating model...")
    grain_model.evaluate_model()


if __name__ == "__main__":
    main()
