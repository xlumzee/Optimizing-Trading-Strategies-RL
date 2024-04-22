import unittest
import pandas as pd
from DataViz_v2 import load_data, plot_histograms, correlation_analysis,  plot_moving_averages, plot_scatter, plot_boxplots, plot_pairplot  # Make sure to import your actual function


class TestDataAnalysis(unittest.TestCase):
    """
    A test file for validating the functions in the DataViz_v2 module which was converted from a notebook to py file.

    This class includes tests that ensure the functions load data correctly, 
    generate plots without errors, and perform data analysis tasks as expected.

    """
    def test_load_data(self):
        """
        Test the load_data function to ensure it correctly loads data from a CSV file into a pandas DataFrame.
        
        Verifies that the function:
        - Returns a DataFrame object.
        - Ensures the DataFrame is not empty.

        """
        df = load_data('/home/amy/work/RIT/TDess/DSCI-601-Amy/Data/Combined/combined_AAPL.csv')  
        self.assertIsInstance(df, pd.DataFrame)  # Check that the result is a DataFrame
        self.assertTrue(not df.empty)  # Ensure that the DataFrame is not empty
        
    def test_plot_histograms(self):
        """
        Test the plot_histograms function to ensure it executes without throwing an error.
        
        Verifies that the function:
        - Successfully executes with a given DataFrame and list of column names.
        - Does not throw any exceptions.

        """
        # ensuring it doesn't throw an error
        try:
            plot_histograms(self.example_df, ['RET', 'VOL_CHANGE'])
            executed = True
        except Exception as e:
            executed = False
        self.assertTrue(executed)

    def test_correlation_analysis(self):
        """
        Test the correlation_analysis function to verify it computes and returns a correlation matrix.
        
        Verifies that the function:
        - Successfully returns a pandas DataFrame.
        - Ensures the DataFrame is a square matrix with dimensions equal to the number of numeric columns in the input DataFrame.
        - Throws an error as we're checking for a 3,3 matrix, but the matrix being plotted is 8,8
        """
        # Ensure the correlation matrix is correct
        try:
            correlation_matrix = correlation_analysis(self.example_df)  # Modify function to return the matrix for testing
            self.assertIsInstance(correlation_matrix, pd.DataFrame)
            self.assertEqual(correlation_matrix.shape, (3, 3))  # Should be square matrix with dimensions equal to number of numeric columns
        except:
            self.fail("Correlation_analysis raised an exception unexpectedly!")

if __name__ == '__main__':
    unittest.main()
