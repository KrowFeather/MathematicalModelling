import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.stattools import kpss
#from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm
import arch.unitroot
import matplotlib.pyplot as plt

def calculate_aic_bic(logL, num_params, num_obs):
    """Calculate AIC and BIC based on log-likelihood."""
    aic = -2 * logL + 2 * num_params
    bic = -2 * logL + num_params * np.log(num_obs)
    return aic, bic

def find_pq(data, pmax, qmax, d):
    """Find optimal p and q based on AIC and BIC criteria."""
    data = np.reshape(data, (len(data), 1))
    LOGL = np.zeros((pmax + 1, qmax + 1))
    PQ = np.zeros((pmax + 1, qmax + 1))
    
    for p in range(pmax + 1):
        for q in range(qmax + 1):
            try:
                model = ARIMA(data, order=(p, d, q))
                fit = model.fit()
                LOGL[p, q] = fit.llf  # Log-likelihood
                PQ[p, q] = p + q      # Number of parameters
            except:
                LOGL[p, q] = -np.inf
    
    # Flatten the arrays
    LOGL_flat = LOGL.flatten()
    PQ_flat = PQ.flatten() + 1
    
    # Calculate AIC and BIC
    aic = np.zeros_like(LOGL_flat)
    bic = np.zeros_like(LOGL_flat)
    
    num_obs = len(data)
    for i in range(len(LOGL_flat)):
        aic[i], bic[i] = calculate_aic_bic(LOGL_flat[i], PQ_flat[i], num_obs)
    
    # Reshape to (pmax+1, qmax+1)
    aic = np.reshape(aic, (pmax + 1, qmax + 1))
    bic = np.reshape(bic, (pmax + 1, qmax + 1))

    # Find the minimum AIC and BIC
    p0, q0 = np.unravel_index(np.argmin(aic), aic.shape)
    p1, q1 = np.unravel_index(np.argmin(bic), bic.shape)

    # Choose p and q based on the criteria in the MATLAB code
    if p0**2 + q0**2 > p1**2 + q1**2:
        p, q = p1, q1
    else:
        p, q = p0, q0
    
    return p, q

def ARIMA_model(data, step):
    """Fit ARIMA model and forecast."""
    ddata = data.copy()
    #print(len(data))
    print(ddata[0])
    d = 0
    #print(ddata)
    #print(a.pvalue)
#    print(a)
    #exit(0)
#    ddata=np.diff(ddata)
    while 1: 
        a = arch.unitroot.KPSS(ddata)
        #_,p_value,_,_=kpss_test(ddata,regression='ct')
        if(a.pvalue<0.05):
            break
        #print(len(ddata))
        ddata = np.diff(ddata)
        print(ddata[0])
        d += 1
        if d > 3:
            break
    print(d)  
    #exit(0)
    pmax, qmax = 3, 3
    p, q = find_pq(data, pmax, qmax, d)
    
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()
    
    forecast_data = model_fit.get_forecast(steps=step)
    forData = forecast_data.predicted_mean
    ymse = forecast_data.se_mean

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(data)), data, label="Historical Data", color='blue')
    plt.plot(range(len(data), len(data) + step), forData, label="Forecasted Data", color='red')
    plt.fill_between(range(len(data), len(data) + step), 
                     forData - 1.96 * ymse, 
                     forData + 1.96 * ymse, color='red', alpha=0.3)
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("ARIMA Model Forecast")
    plt.legend()
    plt.show()
    
    return forData, p, d, q

def kpss_test(x, regression='c', nlags=None):
    """
    Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.

    Parameters:
    x (array_like): The data series.
    regression (str): The type of regression to use. Can be 'c' (constant) or 'ct' (constant and trend).
    nlags (int): Number of lags to use in the Newey-West estimator. If None, it defaults to int(12 * (n / 100)**(1/4)).

    Returns:
    tuple: The test statistic and p-value.
    """
    n = x.shape[0]
    
    if nlags is None:
        nlags = int(12 * (n / 100)**(1/4))
    
    # Calculate the mean of the series
    mean = np.mean(x)
    
    # Calculate the cumulative sum of the series
    cumsum = np.cumsum(x - mean)
    
    # Calculate the long-run variance using the Newey-West estimator
    s = np.zeros(nlags + 1)
    for i in range(nlags + 1):
        s[i] = np.sum((cumsum[i:] - cumsum[:-i])**2)
    s = np.sum(s) / n**2
    
    # Calculate the test statistic
    test_statistic = np.sum((cumsum - np.mean(cumsum))**2) / (n**2 * s)
    
    # Calculate the p-value
    p_value = 1 - norm.cdf(test_statistic, loc=0, scale=1)
    
    return test_statistic, p_value

# Example usage:
data = pd.read_excel('太阳黑子数.xlsx')
#print(data)
#exit(0)
data=data.iloc[-1001:, 3]
#print(data)
#exit(0)
data=data.values

step = 30  # Number of steps to forecast
forData, p, d, q = ARIMA_model(data, step)
print(p,d,q)