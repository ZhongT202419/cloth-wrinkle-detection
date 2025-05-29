import pandas as pd

# Define the file paths
file_paths = {
    'ETF_prices': r'E:\Mutual Fund\archive\ETF prices.csv',
    'ETFs': r'E:\Mutual Fund\archive\ETFs.csv',
    'MutualFund_prices_A_E': r'E:\Mutual Fund\archive\MutualFund prices - A-E.csv',
    'MutualFund_prices_F_K': r'E:\Mutual Fund\archive\MutualFund prices - F-K.csv',
    'MutualFund_prices_L_P': r'E:\Mutual Fund\archive\MutualFund prices - L-P.csv',
    'MutualFund_prices_Q_Z': r'E:\Mutual Fund\archive\MutualFund prices - Q-Z.csv',
    'MutualFunds': r'E:\Mutual Fund\archive\MutualFunds.csv',
    'ie_data': r'E:\Mutual Fund\archive\ie_data.csv',
    'USMacro': r'E:\Mutual Fund\archive\US_MACRO.csv'
}

# Load the CSV files
dataframes = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Display the basic info and first few rows of each file
for name, df in dataframes.items():
    print(f"--- {name} ---")
    print(df.info())  # Structure of the dataset
    print(df.head())  # First few rows
    print("\n")
