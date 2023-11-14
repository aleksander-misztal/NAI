from movie_recommender import MovieRecommender

if __name__ == "__main__":
    """
    Main script to demonstrate the usage of the MovieRecommender class.

    This script initializes a MovieRecommender, trains the collaborative filtering model,
    and then retrieves and prints the top 5 movie recommendations and 5 movies to avoid
    for a specified user.
    """

    # Set the file path for the processed data CSV file
    file_path = 'processed_data.csv'

    # Create an instance of the MovieRecommender class
    recommender = MovieRecommender(file_path)

    # Train the collaborative filtering model
    recommender.train_model()

    # Specify the user ID for which recommendations are to be made
    user_id_to_predict = 2

    # Get the top 5 movie recommendations for the specified user
    top_5_recommendations = recommender.get_top_recommendations(
        user_id_to_predict)

    # Get 5 movies to avoid for the specified user
    bottom_5_avoid = recommender.get_movies_to_avoid(user_id_to_predict)

    # Display top 5 recommended movies for the user
    print(f"Top 5 recommended movies for User {user_id_to_predict}:")
    for movie_id in top_5_recommendations:
        # Extract and print the movie title based on movie ID
        print(recommender.df[recommender.df['movieId']
              == movie_id]['movieId'].values[0])

    # Display 5 movies to avoid for the user
    print("\n5 Movies to Avoid:")
    for movie_id in bottom_5_avoid:
        # Extract and print the movie title based on movie ID
        print(recommender.df[recommender.df['movieId']
              == movie_id]['movieId'].values[0])
