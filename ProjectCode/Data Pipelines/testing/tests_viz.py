import unittest
import pandas as pd
from DataViz_v2 import load_data, plot_histograms, correlation_analysis,  plot_moving_averages, plot_scatter, plot_boxplots, plot_pairplot  # Make sure to import your actual function


class TestDataAnalysis(unittest.TestCase):

    def test_load_data(self):
        df = load_data('/home/amy/work/RIT/TDess/DSCI-601-Amy/Data/Combined/combined_AAPL.csv')  
        self.assertIsInstance(df, pd.DataFrame)  # Check that the result is a DataFrame
        self.assertTrue(not df.empty)  # Ensure that the DataFrame is not empty
        
    def test_plot_histograms(self):
        # ensuring it doesn't throw an error
        try:
            plot_histograms(self.example_df, ['RET', 'VOL_CHANGE'])
            executed = True
        except Exception as e:
            executed = False
        self.assertTrue(executed)

    def test_correlation_analysis(self):
        # Ensure the correlation matrix is correct
        try:
            correlation_matrix = correlation_analysis(self.example_df)  # Modify function to return the matrix for testing
            self.assertIsInstance(correlation_matrix, pd.DataFrame)
            self.assertEqual(correlation_matrix.shape, (3, 3))  # Should be square matrix with dimensions equal to number of numeric columns
        except:
            self.fail("correlation_analysis raised an exception unexpectedly!")

if __name__ == '__main__':
    unittest.main()
