# Description: This file contains the functions that are used in the main file partA.ipynb
from Functions.utils import *

####################################################################################################
####################################################################################################
####################################################################################################
def max_sharpe(df, rf):
    """
    This function calculates the weights of the portfolio that maximizes the Sharpe ratio given past data.
    
    Parameters:
    df : DataFrame
        A DataFrame containing the returns of the assets.
    
    rf : DataFrame
        The risk-free rate. It should have the same length as df.

    Returns:
    w_max_sharpe : array
        The weights of the portfolio that maximizes the Sharpe ratio.
    """

    R = rf.iloc[-1, 0]

    # Calculate the expected returns
    z = np.array(df.mean())
    
    # Calculate the covariance matrix
    S_inv = np.linalg.inv(df.cov())

    # Number of assets
    n = len(z)

    A = np.ones(n) @ S_inv @ np.ones(n)
    B = z @ S_inv @ np.ones(n)

    # Calculate the tangency portfolio weights
    w = (S_inv @ (z - R)) / (B - R * A)

    return w

####################################################################################################
####################################################################################################
####################################################################################################

def max_sharpe_short_sell(df, rf):
    """
    This function calculates the weights of the portfolio that maximizes the Sharpe ratio given past data 
    using numerical optimization and restricting short selling.
    Parameters:
    df : DataFrame
        A DataFrame containing the returns of the assets.
    
    rf : DataFrame
        The risk-free rate. It should have the same length as df.

    Returns:
    w_max_sharpe : array
        The weights of the portfolio that maximizes the Sharpe ratio.
    """

    R = rf.iloc[-1, 0]
    
    # Calculate the expected returns
    z = np.array(df.mean())
        
    # Calculate the covariance matrix
    S = df.cov()
    
    # Number of assets
    n = len(z)
    
    # Define the objective function
    def objective(w):
        return -((z - R) @ w) / np.sqrt(w @ S @ w)
    
    # Define the constraints
    full_invest = LinearConstraint(np.ones(n), 1, 1)
    
    # Initial guess
    w0 = np.ones(n) / n
    
    bounds = [(0, 1) for i in range(n)]

    # Optimize
    res = minimize(objective, w0, constraints= full_invest, bounds = bounds, method='SLSQP')
    
    if res.success:
        return np.array(res.x)
    else:
        raise ValueError('Optimization failed')

####################################################################################################
####################################################################################################
####################################################################################################
    
def inv_var(df):
    """
    Calculate the weights based on an inverse variance strategy.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
    weights (numpy.ndarray): array of weights

    """
    # Calculate the covariance matrix
    S = df.cov()

    # Calculate the inverse variance weights
    inv_var = 1 / np.diag(S)
    w = inv_var / inv_var.sum()

    return w

####################################################################################################
####################################################################################################
####################################################################################################

def inv_vol(df):
    """
    Calculate the weights based on an inverse volatility strategy.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
    weights (numpy.ndarray): array of weights

    """
    # Calculate the covariance matrix
    S = df.cov()

    # Calculate the inverse volatility weights
    inv_vol = 1 / np.sqrt(np.diag(S))
    w = inv_vol / inv_vol.sum()

    return w

####################################################################################################
####################################################################################################
####################################################################################################

def equally_w(df):
    """
    Calculate the weights based on an equal weight strategy.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
    weights (list): Matrix of weights.

    """
    n = len(df.columns)
    w = np.ones(n) / n
    w = [w] * len(df)
    return w

####################################################################################################
####################################################################################################
####################################################################################################

def mrkt_cap(df_cap):
    """
    Calculate the weights based on the market capitalization given the last observation.

    Parameters:
    size (pandas.DataFrame): The industry capitalization.

    Returns:
    weights (numpy.ndarray): array of weights

    """
    total_size = df_cap.iloc[-1].sum()
    w = df_cap.iloc[-1] / total_size
    return w

####################################################################################################
####################################################################################################
####################################################################################################

def min_var(df):
    """
    Calculate the minimum variance portfolio.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
    weights (numpy.ndarray): array of weights

    """
    # Calculate the covariance matrix
    S_inv = np.linalg.inv(df.cov())

    # Number of assets
    n = len(S_inv)

    A = np.ones(n) @ S_inv @ np.ones(n)

    # Calculate the minimum variance weights
    w = S_inv @ np.ones(n) / A

    return w

####################################################################################################
####################################################################################################
####################################################################################################

def performance(weights, returns, rf, start_date, end_date, strat_name, pf_start_val=1):
    """
    Calculate the performance statistics of a portfolio based on the given weights, returns, risk-free rate, start date, and end date.

    Parameters:
    - weights (DataFrame): DataFrame containing the weights of the portfolio assets.
    - returns (DataFrame): DataFrame containing the returns of the portfolio assets.
    - rf (DataFrame): DataFrame containing the risk-free rate.
    - start_date (str): Start date of the analysis period in the format 'YYYY-MM'.
    - end_date (str): End date of the analysis period in the format 'YYYY-MM'.
    - pf_start_val (float, optional): Initial value of the portfolio. Default is 1.

    Returns:
    - results (dict): Dictionary containing the following performance statistics:
        - 'Monthly Returns' (DataFrame): DataFrame containing the monthly returns of the portfolio.
        - 'Portfolio Value' (DataFrame): DataFrame containing the portfolio value over time.
        - 'Statistics' (DataFrame): DataFrame containing the performance statistics.
        - 'Sharpe Ratio' (float): Sharpe ratio of the portfolio.
    """
    # Filter the data based on the specified start and end dates
    weights = weights.loc[start_date:end_date]
    returns = returns.loc[start_date:end_date]
    rf = rf.loc[start_date:end_date]

    # Calculate the portfolio returns
    monthly_returns = (weights * returns).sum(axis=1)
    monthly_returns = pd.DataFrame(monthly_returns, columns=[strat_name], index=returns.index)

    # Calculate the portfolio value over time
    pf_val = (1 + monthly_returns).cumprod() * pf_start_val
    pf_val.columns = [strat_name]

    # Calculate performance statistics
    arith_mean_ann = monthly_returns.mean().values[0] * 12
    geo_mean_ann = (1 + monthly_returns).prod().item() ** (12 / len(monthly_returns)) - 1
    total_return = pf_val.iloc[-1].values[0] - pf_start_val
    std_ann = monthly_returns.std().values[0] * np.sqrt(12)
    min_return = monthly_returns.min().values[0]
    max_return = monthly_returns.max().values[0]
    sharpe_ratio = (monthly_returns.mean().values[0] - rf.mean().values[0]) / monthly_returns.std().values[0] * np.sqrt(12)

    # Create a dictionary to store the performance statistics
    stats = {'Arithmetic Mean (%)': round(arith_mean_ann * 100, 2),
             'Geometric Mean (%)': round(geo_mean_ann * 100, 2) if isinstance(geo_mean_ann, complex) == False else 'N/A',
             'Total Return (%)': round(total_return * 100, 2),
             'Portfolio Value ($)': round(pf_val.iloc[-1].values[0], 2),
             'Standard Deviation (%)': round(std_ann * 100, 2),
             'Min Return (monthly) (%)': round(min_return * 100, 2),
             'Max Return (monthly) (%)': round(max_return * 100, 2),
             'Sharpe Ratio': round(sharpe_ratio, 4),
             }

    # Convert the dictionary to a DataFrame
    stats = pd.DataFrame(stats, index=[strat_name])

    # Create a dictionary to store the results
    results = {'Monthly Returns': monthly_returns, 
               'Portfolio Value': pf_val, 
               'Statistics': stats}

    return results

####################################################################################################
####################################################################################################
####################################################################################################

def plot_returns(monthly_returns, strat_name, lim = None, path=None):
        """
        Plot the density distribution of monthly returns.

        Parameters:
        - monthly_returns (DataFrame): DataFrame containing the monthly returns data.
        - path (str, optional): Path to save the plot. If not provided, the plot will be displayed.

        Returns:
        None
        """
        # Plot the density distribution of monthly returns
        monthly_returns.plot(kind='kde', color='darkblue', figsize=(12, 6))

        # Add mean and median lines
        mean_return = monthly_returns.mean().values[0]
        median_return = monthly_returns.median().values[0]
        plt.axvline(mean_return, color='r', linestyle='--', label='Mean')
        plt.axvline(median_return, color='g', linestyle='--', label='Median')
        plt.grid(color='lightgray', linestyle='--')
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=2))
        plt.xlim(-lim,lim)

        # Add labels and legend
        plt.title('Density Distribution of Monthly Returns: ' + strat_name)
        plt.xlabel('Monthly Returns')
        plt.ylabel('Density')
        plt.legend()

        if path:
                plt.savefig(path, bbox_inches='tight')
        else:
                plt.show()
        
        return None

####################################################################################################
####################################################################################################
####################################################################################################

def plot_pf_val(pf_val, strat_name, path=None):
        """
        Plot the portfolio value over time.

        Parameters:
        - pf_val (DataFrame): DataFrame containing the portfolio value over time.
        - path (str, optional): Path to save the plot. If not provided, the plot will be displayed.

        Returns:
        None
        """
        # Plot the portfolio value over time
        plt.figure(figsize=(12, 6))
        pf_val.plot(color='darkblue')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.title('Portfolio Value Over Time' + ' (' + strat_name + ')')
        plt.grid(color='lightgray', linestyle='--')
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.2f}'))
        if path:
                plt.savefig(path, bbox_inches='tight')
        else:
                plt.show()
                
        return None