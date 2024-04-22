import unittest
import pandas as pd
from DataPrepv2 import load_data, create_lagged_features, calculate_rolling_statistics  # Adjust the import according to your setup

class TestDataFunctions(unittest.TestCase):

    def test_load_data_success(self):
        """ Test loading of data and conversion of 'date' column to datetime. """
        df = load_data('/home/amy/work/RIT/TDess/DSCI-601-Amy/Data/Combined/combined_AAPL.csv')  # Ensure this test file exists and is formatted correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['date']))

    def test_load_data_file_not_found(self):
        """ Test the function raises the correct exception when file is not found. """
        with self.assertRaises(FileNotFoundError):
            load_data('non_existent_file.csv')

# add tests for lagged features and rolling stats

if __name__ == '__main__':
    unittest.main()

