# Regression Analysis with Statsmodels

Below is an anlysis of the `brainsize.csv` data found in this repo.  


```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
#import data
brain = pd.read_table("brainsize.csv", sep = ";", index_col=0, na_values = ".")

brain.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>FSIQ</th>
      <th>VIQ</th>
      <th>PIQ</th>
      <th>Weight</th>
      <th>Height</th>
      <th>MRI_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>133</td>
      <td>132</td>
      <td>124</td>
      <td>118.0</td>
      <td>64.5</td>
      <td>816932</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>140</td>
      <td>150</td>
      <td>124</td>
      <td>NaN</td>
      <td>72.5</td>
      <td>1001121</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>139</td>
      <td>123</td>
      <td>150</td>
      <td>143.0</td>
      <td>73.3</td>
      <td>1038437</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>133</td>
      <td>129</td>
      <td>128</td>
      <td>172.0</td>
      <td>68.8</td>
      <td>965353</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Female</td>
      <td>137</td>
      <td>132</td>
      <td>134</td>
      <td>147.0</td>
      <td>65.0</td>
      <td>951545</td>
    </tr>
  </tbody>
</table>
</div>



The brainsize data includes basic demographic variables as well as several measures of intelligence.


```python
np.random.seed(242)
#crete new variable
partY = np.random.randn(len(brain))

#add new var to data
brain["partY"] = partY

```


```python
brain.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>FSIQ</th>
      <th>VIQ</th>
      <th>PIQ</th>
      <th>Weight</th>
      <th>Height</th>
      <th>MRI_Count</th>
      <th>partY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>133</td>
      <td>132</td>
      <td>124</td>
      <td>118.0</td>
      <td>64.5</td>
      <td>816932</td>
      <td>-0.357519</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>140</td>
      <td>150</td>
      <td>124</td>
      <td>NaN</td>
      <td>72.5</td>
      <td>1001121</td>
      <td>0.148448</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>139</td>
      <td>123</td>
      <td>150</td>
      <td>143.0</td>
      <td>73.3</td>
      <td>1038437</td>
      <td>0.993531</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>133</td>
      <td>129</td>
      <td>128</td>
      <td>172.0</td>
      <td>68.8</td>
      <td>965353</td>
      <td>1.838968</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Female</td>
      <td>137</td>
      <td>132</td>
      <td>134</td>
      <td>147.0</td>
      <td>65.0</td>
      <td>951545</td>
      <td>-0.744026</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot partY var as a function of FSIQ and VIQ
plt.figure(figsize=(10, 6))
plt.scatter(x = brain["PIQ"], y = brain["partY"], s = brain["FSIQ"]*4, alpha = 0.5, label = "FSIQ")
plt.title("partY as a function of PIQ and FSIQ")
plt.xlabel("PIQ")
plt.ylabel("partY")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f91b4b2d110>




![png](myanalysis_files/myanalysis_7_1.png)


The partY variable seems to increase with FSIQ and PIQ.


```python
# create linear model with interaction only
from statsmodels.formula.api import ols
model1 = ols("partY ~ FSIQ:PIQ", brain).fit()
print(model1.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  partY   R-squared:                       0.170
    Model:                            OLS   Adj. R-squared:                  0.148
    Method:                 Least Squares   F-statistic:                     7.764
    Date:                Fri, 29 May 2020   Prob (F-statistic):            0.00827
    Time:                        17:08:57   Log-Likelihood:                -50.962
    No. Observations:                  40   AIC:                             105.9
    Df Residuals:                      38   BIC:                             109.3
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -0.8531      0.392     -2.174      0.036      -1.648      -0.059
    FSIQ:PIQ    7.803e-05    2.8e-05      2.786      0.008    2.13e-05       0.000
    ==============================================================================
    Omnibus:                        1.332   Durbin-Watson:                   2.090
    Prob(Omnibus):                  0.514   Jarque-Bera (JB):                1.202
    Skew:                           0.402   Prob(JB):                        0.548
    Kurtosis:                       2.729   Cond. No.                     3.92e+04
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 3.92e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.


The interaction between FSIQ and PIQ was a significant predictor of partY (*b*=0.00007, *t*=2.79, *p*=.008). The overall model significantly predicted partY (F(1,38)=7.76, *p*=.008, $R^{2}$=.17).


```python
np.random.seed(312)
#crete second random variable
partY2 = np.random.randn(len(brain))

#add new var to data
brain["partY2"] = partY2

```


```python
model2 = ols("partY2 ~ FSIQ:PIQ", brain).fit()
print(model2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 partY2   R-squared:                       0.027
    Model:                            OLS   Adj. R-squared:                  0.001
    Method:                 Least Squares   F-statistic:                     1.037
    Date:                Fri, 29 May 2020   Prob (F-statistic):              0.315
    Time:                        17:08:57   Log-Likelihood:                -55.477
    No. Observations:                  40   AIC:                             115.0
    Df Residuals:                      38   BIC:                             118.3
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.3261      0.439      0.742      0.463      -0.563       1.216
    FSIQ:PIQ   -3.192e-05   3.14e-05     -1.018      0.315   -9.54e-05    3.15e-05
    ==============================================================================
    Omnibus:                        0.360   Durbin-Watson:                   2.108
    Prob(Omnibus):                  0.835   Jarque-Bera (JB):                0.491
    Skew:                          -0.192   Prob(JB):                        0.783
    Kurtosis:                       2.618   Cond. No.                     3.92e+04
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 3.92e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.


The interaction of FSIQ and PIQ did not significantly predict partY2 (F(1,38)=1.01, *p*=.32, $R^{2}$=.03).
