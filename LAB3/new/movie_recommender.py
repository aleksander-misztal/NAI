import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split


class MovieRecommender:
    def __init__(self, file_path):
        """
        Initialize the MovieRecommender class.

        Parameters:
        - file_path (str): The path to the CSV file containing user ratings.
        """
        # Read data from CSV file into a Pandas DataFrame
        self.df = pd.read_csv(file_path)

        # Configure the rating scale for Surprise library
        self.reader = Reader(rating_scale=(1, 10))

        # Load the data into Surprise Dataset
        self.data = Dataset.load_from_df(
            self.df[['userId', 'movieId', 'rating']], self.reader)

        # Split the data into training and testing sets
        self.trainset, self.testset = train_test_split(
            self.data, test_size=0.2, random_state=42)

        # Initialize the KNNBasic collaborative filtering model
        self.model = KNNBasic(
            sim_options={'name': 'cosine', 'user_based': False})

    def train_model(self):
        """
        Train the collaborative filtering model using the training set.
        """
        self.model.fit(self.trainset)

    def get_top_recommendations(self, user_id, top_n=5):
        """
        Get top movie recommendations for a given user.

        Parameters:
        - user_id (int): The ID of the user for whom recommendations are needed.
        - top_n (int): The number of top recommendations to retrieve (default is 5).

        Returns:
        - list: A list of movie IDs recommended for the user.
        """
        user_movies = self.df[self.df['userId'] == user_id]['movieId'].tolist()
        movies_to_predict = [
            movie for movie in self.df['movieId'].unique() if movie not in user_movies]
        predictions = [self.model.predict(user_id, movie)
                       for movie in movies_to_predict]
        sorted_predictions = sorted(
            predictions, key=lambda x: x.est, reverse=True)
        top_recommendations = [pred.iid for pred in sorted_predictions[:top_n]]
        return top_recommendations

    def get_movies_to_avoid(self, user_id, bottom_n=5):
        """
        Get movies to avoid for a given user.

        Parameters:
        - user_id (int): The ID of the user for whom movies to avoid are needed.
        - bottom_n (int): The number of bottom-ranked movies to avoid (default is 5).

        Returns:
        - list: A list of movie IDs to avoid for the user.
        """
        user_movies = self.df[self.df['userId'] == user_id]['movieId'].tolist()
        movies_to_predict = [
            movie for movie in self.df['movieId'].unique() if movie not in user_movies]
        predictions = [self.model.predict(user_id, movie)
                       for movie in movies_to_predict]
        sorted_predictions = sorted(
            predictions, key=lambda x: x.est, reverse=True)
        movies_to_avoid = [pred.iid for pred in sorted_predictions[-bottom_n:]]
        return movies_to_avoid
