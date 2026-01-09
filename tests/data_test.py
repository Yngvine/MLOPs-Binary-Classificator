import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
from custom_lib.data import load_data, plot_class_distribution

class TestDataLoading(unittest.TestCase):
    @patch('custom_lib.data.pd.read_csv')
    @patch('custom_lib.data.UNPACKED_DIR', Path('/mock/dir'))
    def test_load_data_default(self, mock_read_csv):
        """Test load_data with default path."""
        mock_df = pd.DataFrame({'a': [1, 2]})
        mock_read_csv.return_value = mock_df
        
        df = load_data()
        
        # Check if read_csv was called with expected path
        # Note: on Windows /mock/dir might need handling, but we mocked UNPACKED_DIR
        expected_path = Path('/mock/dir/riceClassification.csv')
        mock_read_csv.assert_called_once_with(expected_path)
        self.assertEqual(df.shape, (2, 1))

    @patch('custom_lib.data.pd.read_csv')
    def test_load_data_custom_path(self, mock_read_csv):
        """Test load_data with provided path."""
        load_data("custom.csv")
        mock_read_csv.assert_called_once_with("custom.csv")

class TestDataPlotting(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'Class': [0, 1, 0, 1, 0],
            'Area': [100, 200, 150, 250, 120]
        })

    @patch('custom_lib.data.plt')
    @patch('custom_lib.data.sns')
    def test_plot_class_distribution(self, mock_sns, mock_plt):
        """Test that plot_class_distribution calls seaborn and returns figure."""
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, MagicMock())
        
        # Test without save
        fig = plot_class_distribution(self.df)
        
        self.assertEqual(fig, mock_fig)
        mock_sns.barplot.assert_called_once()
        
        # Test with save
        fig = plot_class_distribution(self.df, save_path="plot.png")
        mock_fig.savefig.assert_called_once()

