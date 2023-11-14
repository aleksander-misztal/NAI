import pandas as pd

class RecommendationEngine:
    def __init__(self, data_file):
        """
        Initialize the RecommendationEngine with data from a CSV file.

        Parameters:
        - data_file (str): Path to the CSV file containing user movie ratings.
        """
        # Read the CSV file into a DataFrame
        self.df = pd.read_csv(data_file)

    def _filter_user(self, user):
        """
        Filter the DataFrame to exclude a specified user.

        Parameters:
        - user (str): The user to be excluded.

        Returns:
        - pd.DataFrame: DataFrame with the specified user excluded.
        """
        # Check if the specified user exists in the DataFrame
        if user not in self.df['User'].values:
            return pd.DataFrame()

        # Filter the DataFrame to exclude the specified user
        return self.df[self.df['User'] != user]

    def _create_user_movie_matrix(self, other_users):
        """
        Create a user-movie matrix based on the ratings of other users.

        Parameters:
        - other_users (pd.DataFrame): DataFrame excluding the target user.

        Returns:
        - pd.DataFrame: User-movie matrix.
        """
        # Create a user-movie matrix
        return other_users.pivot_table(index='User', columns='Movie1', values='Grade1', fill_value=0)

    def _get_movie_ratings_count(self, user_movie_matrix):
        """
        Calculate the number of ratings for each movie in the user-movie matrix.

        Parameters:
        - user_movie_matrix (pd.DataFrame): User-movie matrix.

        Returns:
        - pd.Series: Series with the count of ratings for each movie.
        """
        # Calculate the number of ratings for each movie
        return user_movie_matrix.astype(bool).sum(axis=0)

    def recommend_movies(self, user):
        """
        Recommend movies for a given user based on popularity.

        Parameters:
        - user (str): The target user.

        Returns:
        - tuple: A tuple containing two lists:
            - List of recommended movies.
            - List of not recommended movies.
        """
        # Filter the DataFrame to exclude the specified user
        other_users = self._filter_user(user)

        # Check if there are other users in the DataFrame
        if other_users.empty:
            return [], []

        # Create a user-movie matrix
        user_movie_matrix = self._create_user_movie_matrix(other_users)

        # Check if the user has not watched any movies
        if user_movie_matrix.empty:
            return [], []

        # Calculate the number of ratings for each movie
        movie_ratings_count = self._get_movie_ratings_count(user_movie_matrix)

        # Get the top 5 most popular movies
        popular_movies = movie_ratings_count.sort_values(
            ascending=False).index[:5]

        # Find the movies that the target user has not watched yet
        target_user_movies = self.df[self.df['User']
                                     == user].iloc[:, 1::2].replace(0, pd.NA)

        # Get the popular movies that the target user has not watched
        recommended_movies = [
            movie for movie in popular_movies if movie not in target_user_movies.columns]

        # Get the 5 least popular movies that the target user has not watched
        not_recommended_movies = movie_ratings_count.sort_values(
        ).index[:5].tolist()

        return recommended_movies, not_recommended_movies
