from Functions.utils import *

Figures_PATH = Path.cwd() / 'Figures'

def get_betas(rets, col, indu, ff_factors): 
    """Get the betas of the industry portfolio using a OLS regression

    Args:
        rets (pd.Series): The returns of the industry portfolios
        col (str): The name of the column in the industry portfolio dataframe

    Returns:
        pd.Series: The betas of the industry portfolio
    """
    betas = {} # dictionary to store the betas
    for row in indu.iterrows():
        month = row[0] # get the month

        # get the returns of the industry portfolio for the past 12 months
        y = rets[month + pd.offsets.MonthEnd(-12):month][col]
        x = sm.add_constant(ff_factors[month + pd.offsets.MonthEnd(-12):month]['Mkt-RF'])

        # fit the OLS regression
        rols = sm.regression.linear_model.OLS(y, x)
        rres = rols.fit()

        # get the beta and store it in the dictionary
        params = rres.params.iloc[1]
        betas[month] = params

    return pd.Series(betas)

def get_idio_vol(rets, col, indu, ff_factors): 
    """Get the idiovols of the industry portfolio using a OLS regression

    Args:
        rets (pd.Series): The returns of the industry portfolios
        col (str): The name of the column in the industry portfolio dataframe

    Returns:
        pd.Series: The idiovols of the industry portfolio
    """
    idio_vols = {} # dictionary to store the idiovols
    for row in indu.iterrows():
        month = row[0] # get the month
 
        # get the returns of the industry portfolio for the past 12 months
        y = rets[month + pd.offsets.MonthEnd(-12):month][col]
        x = sm.add_constant(ff_factors[month + pd.offsets.MonthEnd(-12):month].drop(columns=['RF']))

        # fit the OLS regression
        rols = sm.regression.linear_model.OLS(y, x)
        rres = rols.fit()

        # get the residuals and store the idiovol in the dictionary
        residuals = rres.resid
        idio_vols[month] = residuals.std() # idiovol is the standard deviation of the residuals

    return pd.Series(idio_vols)


def get_cum_rets(df):
    return (1 + df).cumprod() - 1


def plot_portfolios(ew_df, vw_df, ff_factors, four_ff_factors, five_ff_factors, start, end):
    """Plot the factor portfolios

    Args:
        charac_df (df): The dataframe containing the returns of the characteristic portfolios
        ff_factors (df): fama french three factors df
        four_ff_factors (df): fama french four factors df
        five_ff_factors (df): fama french five factors df
        start (str): period start
        end (str): period end
    """
    plt.subplots(figsize=(20, 10), nrows=3, ncols=3) # create the figure

    # plot the market factor
    plt.subplot(3, 3, 1)
    plt.plot(get_cum_rets(ff_factors['Mkt-RF'][start:end]))
    plt.title(f'Mkt-RF Factor Total Return {start}-{end}')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.grid(True)

    # plot the idiovol factor
    plt.subplot(3, 3, 2)
    plt.plot(get_cum_rets(ew_df['Idiovol'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Equally-Weighted Idiovol')
    plt.plot(get_cum_rets(vw_df['Idiovol'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Value-Weighted Idiovol')
    plt.title(f'Idiovol Factor Total Return {start}-{end}')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.legend()
    plt.grid(True)

    # plot the beta factor
    plt.subplot(3, 3, 3)
    plt.plot(get_cum_rets(ew_df['Beta'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Equally-Weighted Beta')
    plt.plot(get_cum_rets(vw_df['Beta'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Value-Weighted Beta')
    plt.title(f'Beta Factor Total Return {start}-{end}')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.legend()
    plt.grid(True)

    # plot the size factors
    plt.subplot(3, 3, 4)
    plt.plot(get_cum_rets(ew_df['Size'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Equally-Weighted Size')
    plt.plot(get_cum_rets(vw_df['Size'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Value-Weighted Size')
    plt.plot(get_cum_rets(ff_factors['SMB'][start:end]), label='SMB F-F Three-Factor')
    plt.plot(get_cum_rets(five_ff_factors['SMB'][start:end]), label='SMB F-F Five-Factor')
    plt.title(f'Size Factor Total Return {start}-{end}')
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.legend()
    
    # plot the value factors
    plt.subplot(3, 3, 5)
    plt.plot(get_cum_rets(ew_df['BM'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Equally-Weighted BM')
    plt.plot(get_cum_rets(vw_df['BM'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Value-Weighted BM')
    plt.plot(get_cum_rets(ff_factors['HML'][start:end]), label='HML')
    plt.title(f'Value Factor Total Return {start}-{end}')
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.legend()

    # plot the momentum factors
    plt.subplot(3, 3, 6)
    plt.plot(get_cum_rets(ew_df['Mom'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Equally-Weighted Mom')
    plt.plot(get_cum_rets(vw_df['Mom'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Value-Weighted Mom')
    plt.plot(get_cum_rets(four_ff_factors['Mom   '][start:end]), label='FF-Mom')
    plt.title(f'Momentum Factor Total Return {start}-{end}')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.legend()
    plt.grid(True)

    # plot the profitability and Investment factors
    plt.subplot(3, 3, 7)
    plt.plot(get_cum_rets(five_ff_factors['RMW'][start:end]), label='RMW')
    plt.plot(get_cum_rets(five_ff_factors['CMA'][start:end]), label='CMA')
    plt.title(f'Profitability and Investment Factor Total Return {start}-{end}')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.grid(True)
    plt.legend() 

    # plot the total return of the equally-weighted factor portfolios
    plt.subplot(3, 3, 8)
    plt.plot(get_cum_rets(ew_df['Mom'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Mom')
    plt.plot(get_cum_rets(ew_df['Size'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Size')
    plt.plot(get_cum_rets(ew_df['Beta'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Beta')
    plt.plot(get_cum_rets(ew_df['Idiovol'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Idiovol')
    plt.plot(get_cum_rets(ew_df['BM'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='BM')
    plt.title(f'Total Return of Equally-Weighted Factor Portfolios, {start}-{end}')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.grid(True)
    plt.legend()

    # plot the total return of the value-weighted factor portfolios
    plt.subplot(3, 3, 9)
    plt.plot(get_cum_rets(vw_df['Mom'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Mom')
    plt.plot(get_cum_rets(vw_df['Size'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Size')
    plt.plot(get_cum_rets(vw_df['Beta'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Beta')
    plt.plot(get_cum_rets(vw_df['Idiovol'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='Idiovol')
    plt.plot(get_cum_rets(vw_df['BM'][start:end].sub(four_ff_factors['RF'].loc[start:end], axis=0)), label='BM')
    plt.title(f'Total Return of Value-Weighted Factor Portfolios, {start}-{end}')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(Figures_PATH / 'Part B' / f'Factor_Portfolios{start}_{end}.png') # save the figure



def get_stats(df, start, end, charac, *args):
    """Get the Sharpe ratio and alphas of the characteristic portfolios"""
    ff_factors = args[0]
    four_ff_factors = args[1]
    five_ff_factors = args[2]

    # Calculate the Sharpe ratio
    excess_rets = df.loc[start:end][charac] - ff_factors['RF'].loc[start:end]
    sharpe = round(((excess_rets.mean()) / excess_rets.std()) * np.sqrt(12), 4)

    # Calculate the alphas using the 3, 4 and 5 factor models
    y = excess_rets
    ff_factors_excess_ret = ff_factors.loc[start:end].drop('RF', axis=1)
    four_ff_factors_excess_ret = four_ff_factors.loc[start:end].drop('RF', axis=1)
    five_ff_factors_excess_ret = five_ff_factors.loc[start:end].drop('RF', axis=1)

    x3 = sm.add_constant(ff_factors_excess_ret)
    x4 = sm.add_constant(four_ff_factors_excess_ret[start:end])

    # We adjust the start date for the 5 factor model because it starts in 1963
    if start < '1963-07-31':
        start = '1963-07-31'
        x5 = sm.add_constant(five_ff_factors_excess_ret[start:end])
        y5 = df[start:end][charac] - ff_factors['RF'].loc[start:end]
        ols5 = sm.regression.linear_model.OLS(y5, x5)
    else:
        x5 = sm.add_constant(five_ff_factors_excess_ret)
        ols5 = sm.regression.linear_model.OLS(y, x5)
    
    # Fit the OLS regressions
    ols3 = sm.regression.linear_model.OLS(y, x3)
    ols4 = sm.regression.linear_model.OLS(y, x4)
    res3 = ols3.fit()
    res4 = ols4.fit()
    res5 = ols5.fit()

    # Get the alphas and annualize them
    alpha4 = round(res4.params['const'] * 100, 4) * 12
    alpha5 = round(res5.params['const'] * 100, 4) * 12
    alpha3 = round(res3.params['const'] * 100, 4) * 12 

    return (sharpe, alpha3, alpha4, alpha5)