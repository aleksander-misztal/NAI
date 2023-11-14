# Example usage
from recommendation_engine import RecommendationEngine


if __name__ == '__main__':
    
    # Data path 
    data_file = 'data.csv'

    # Create an instance of the RecommendationEngine.
    movie_recommender = RecommendationEngine(data_file)

    # Specify the user for whom you want to generate recommendations.
    user_to_recommend = "User1"

    # Get the recommended and not recommended movies for the specified user.
    recommended, not_recommended = movie_recommender.recommend_movies(
        user_to_recommend)

    # Print the results.
    print(f"\nRecommended movies for {user_to_recommend}:")
    print(recommended)

    print(f"\nMovies to avoid for {user_to_recommend}:")
    print(not_recommended)
