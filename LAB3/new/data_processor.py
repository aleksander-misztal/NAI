import pandas as pd


class DataProcessor:
    def __init__(self, file_path):
        """
        Initialize the DataProcessor class with the given file path.

        Parameters:
        - file_path (str): The path to the CSV file containing raw data.
        """
        self.file_path = file_path

    def load_data(self):
        """
        Load the raw data from the CSV file.

        Returns:
        - pd.DataFrame: The loaded DataFrame containing raw data.
        """
        return pd.read_csv(self.file_path)

    def convert_to_nested_structure(self, df):
        """
        Convert the raw data DataFrame to a nested structure.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing raw data.

        Returns:
        - list: A list of dictionaries representing nested user data.
        """
        new_data = []
        for _, row in df.iterrows():
            user_id = row['User']
            user_data = {'userId': user_id, 'ratings': []}

            for i in range(1, 32, 2):
                movie_name = row[f'Movie{i}']
                rating = row[f'Grade{i}']

                if not pd.isna(movie_name) and not pd.isna(rating):
                    user_data['ratings'].append(
                        {'movieId': movie_name, 'rating': rating})

            new_data.append(user_data)
        return new_data

    def flatten_data(self, nested_data):
        """
        Flatten the nested user data to a list of dictionaries.

        Parameters:
        - nested_data (list): A list of dictionaries representing nested user data.

        Returns:
        - list: A list of dictionaries containing flattened user data.
        """
        flat_data = []
        for user_data in nested_data:
            for rating in user_data['ratings']:
                flat_data.append(
                    {'userId': user_data['userId'], 'movieId': rating['movieId'], 'rating': rating['rating']})
        return flat_data

    def process_data(self):
        """
        Process the raw data and return a DataFrame with flattened user data.

        Returns:
        - pd.DataFrame: The processed DataFrame with flattened user data.
        """
        df = self.load_data()
        nested_data = self.convert_to_nested_structure(df)
        flat_data = self.flatten_data(nested_data)
        processed_df = pd.DataFrame(flat_data)
        processed_df.to_csv('processed_data.csv', index=False)
        return processed_df


if __name__ == "__main__":
    # Set the file path for the raw data CSV file
    file_path = 'old_data.csv'

    # Create an instance of the DataProcessor class
    processor = DataProcessor(file_path)

    # Process the data and get the processed DataFrame
    processed_df = processor.process_data()

    # Display the processed DataFrame
    print(processed_df)
