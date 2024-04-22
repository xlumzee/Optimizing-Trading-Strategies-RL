# DSCI-601-Amy
Repo to track Capstone 601 - 602 progress

## Dataset


### About the data

| Variable   | Description |
|------------|-------------|
| Date       | Timestamp for the data. |
| RET        | Return on a stock over a given period of time. |
| Vol_Change | Changes in trading volume. Increase in volume can signify high interest, while a decrease might suggest less interest. |
| BA_Spread  | Difference between the highest price a buyer is willing to pay (bid) and the lowest price a seller is willing to accept (ask). A narrower spread often indicates a more liquid market or higher market efficiency, whereas a wider spread can indicate lower liquidity or higher risk. |
| Illiquidity| A measure of the difficulty of trading a stock without affecting its price. High illiquidity means the stock is not easily tradable without significant price changes, which can increase the cost of trading and the risk. |
| Sprtrn     | Return of the S&P 500 index, which is a market-capitalization-weighted index of the 500 largest U.S. publicly traded companies. The S&P 500 is a common benchmark for U.S. stock performance. |
| Turnover   | Refers to the total volume of shares traded during a specific period divided by the total shares outstanding. High turnover can indicate high trading activity, suggesting interest or volatility in the stock. |
| Dji_Return | Return on the Dow Jones Industrial Average, another major stock market index in the United States. It consists of 30 large, publicly-owned companies based in the United States. |

##### The data files are available in the Data folder. After cloning the repository, you can load the data using pandas read_csv function. 



## Data Visualization

DataViz_v2 notebook is available in Data pipelines folder. This notebook has functions that can be reused as they are standardized. They visualize basic plots that show us information about the data. The plots that are used are :
- Histograms
- Correlation Analysis
- Moving Averages
- Scatter Plots
- Pair plots
- Box Plots


## Data Preprocessing and Feature Engineering

The Data preprocessing and feature engineering notebook is present in the Data Pipelines folder. It has the following functions :
- Load data
- Process date column
- create lagged features
- calculate rolling statistics

## Test Cases

There is a testing folder which has the *tests_viz.py* file. This has test cases which run on *DataViz_v2.py*. They verify if the data is being loaded correctly and if the visualizations are being plotted and not throwing any errors. 

The same folder has *tests_DataPrepv2.py* file. This will run on the *DataPrepv2.py*. It has 2 test cases which ensure that the data is loading and some initial preprocessing is being done like the date column being converted to datetime. Need to add for lagged features and rolling statistics.

## Data Modeling 
To be continued....
