import pandas as pd

# Read the data from 'Data/10_Industry_Portfolios.CSV' file, skipping the first 11 rows and reading 1171 rows
data_10_industry = pd.read_csv('Data/10_Industry_Portfolios.CSV', skiprows=11, nrows=1171) # Average Value Weighted Returns -- Monthly

# Rename the column 'Unnamed: 0' to 'Date'
data_10_industry.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# Convert the 'Date' column to datetime format and format it as 'YYYY-MM'
data_10_industry['Date'] = pd.to_datetime(data_10_industry['Date'], format='%Y%m').dt.strftime('%Y-%m')

# Set the 'Date' column as the index
data_10_industry.set_index('Date', inplace=True)

# Divide each value in the DataFrame by 100 and round it to 6 decimal places
data_10_industry = data_10_industry.apply(lambda x: x / 100).round(6)

# Save the cleaned data to 'Data/10_Industry_Portfolios_clean.csv' file
data_10_industry.to_csv('Data/10_Industry_Portfolios_clean.csv', index=True)

##############################################################################################
##############################################################################################
##############################################################################################

# Read the data from 'Data/F-F_Research_Data_Factors.CSV' file, skipping the first 3 rows and reading 1171 rows
risk_free_asset = pd.read_csv('Data/F-F_Research_Data_Factors.CSV', skiprows=3, nrows=1171)

# Rename the column 'Unnamed: 0' to 'Date'
risk_free_asset.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# Convert the 'Date' column to datetime format and format it as 'YYYY-MM'
risk_free_asset['Date'] = pd.to_datetime(risk_free_asset['Date'], format='%Y%m').dt.strftime('%Y-%m')

# Set the 'Date' column as the index
risk_free_asset.set_index('Date', inplace=True)

# Divide each value in the DataFrame by 100 and round it to 6 decimal places
risk_free_asset = risk_free_asset.apply(lambda x: x / 100).round(6)

# Keep only the 'RF' column in the DataFrame
risk_free_asset = risk_free_asset[['RF']]

# Save the cleaned data to 'Data/Risk_Free_Asset_Monthly.csv' file
risk_free_asset.to_csv('Data/Risk_Free_Asset_Monthly.csv', index=True)

##############################################################################################
##############################################################################################
##############################################################################################

# Read the data from the CSV file into a DataFrame
firm_size = pd.read_csv('Data/10_Industry_Portfolios.CSV', skiprows=3738, nrows=1171)

# Rename the column 'Unnamed: 0' to 'Date'
firm_size.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# Convert the 'Date' column to datetime format and format it as 'YYYY-MM'
firm_size['Date'] = pd.to_datetime(firm_size['Date'], format='%Y%m').dt.strftime('%Y-%m')

# Set the 'Date' column as the index
firm_size.set_index('Date', inplace=True)

# Save the cleaned data to 'Data/Firm_Size_10_Industry.csv' file
firm_size.to_csv('Data/Firm_Size_10_Industry.csv', index=True)

##############################################################################################
##############################################################################################
##############################################################################################

# Read the data from the CSV file into a DataFrame
firm_num = pd.read_csv('Data/10_Industry_Portfolios.CSV', skiprows=2563, nrows=1171)

# Rename the column 'Unnamed: 0' to 'Date'
firm_num.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# Convert the 'Date' column to datetime format and format it as 'YYYY-MM'
firm_num['Date'] = pd.to_datetime(firm_num['Date'], format='%Y%m').dt.strftime('%Y-%m')

# Set the 'Date' column as the index
firm_num.set_index('Date', inplace=True)

# Save the cleaned data to 'Data/Firm_Number_10_Industry.csv' file
firm_num.to_csv('Data/Firm_Number_10_Industry.csv', index=True)
