{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Regression Analysis with Statsmodels"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Below is an anlysis of the `brainsize.csv` data found in this repo.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>FSIQ</th>\n",
       "      <th>VIQ</th>\n",
       "      <th>PIQ</th>\n",
       "      <th>Weight</th>\n",
       "      <th>Height</th>\n",
       "      <th>MRI_Count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Female</td>\n",
       "      <td>133</td>\n",
       "      <td>132</td>\n",
       "      <td>124</td>\n",
       "      <td>118.0</td>\n",
       "      <td>64.5</td>\n",
       "      <td>816932</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Male</td>\n",
       "      <td>140</td>\n",
       "      <td>150</td>\n",
       "      <td>124</td>\n",
       "      <td>NaN</td>\n",
       "      <td>72.5</td>\n",
       "      <td>1001121</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Male</td>\n",
       "      <td>139</td>\n",
       "      <td>123</td>\n",
       "      <td>150</td>\n",
       "      <td>143.0</td>\n",
       "      <td>73.3</td>\n",
       "      <td>1038437</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Male</td>\n",
       "      <td>133</td>\n",
       "      <td>129</td>\n",
       "      <td>128</td>\n",
       "      <td>172.0</td>\n",
       "      <td>68.8</td>\n",
       "      <td>965353</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Female</td>\n",
       "      <td>137</td>\n",
       "      <td>132</td>\n",
       "      <td>134</td>\n",
       "      <td>147.0</td>\n",
       "      <td>65.0</td>\n",
       "      <td>951545</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Gender  FSIQ  VIQ  PIQ  Weight  Height  MRI_Count\n",
       "1  Female   133  132  124   118.0    64.5     816932\n",
       "2    Male   140  150  124     NaN    72.5    1001121\n",
       "3    Male   139  123  150   143.0    73.3    1038437\n",
       "4    Male   133  129  128   172.0    68.8     965353\n",
       "5  Female   137  132  134   147.0    65.0     951545"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#import data\n",
    "brain = pd.read_table(\"brainsize.csv\", sep = \";\", index_col=0, na_values = \".\")\n",
    "\n",
    "brain.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The brainsize data includes basic demographic variables as well as several measures of intelligence."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.random.seed(242)\n",
    "#crete new variable\n",
    "partY = np.random.randn(len(brain))\n",
    "\n",
    "#add new var to data\n",
    "brain[\"partY\"] = partY\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>FSIQ</th>\n",
       "      <th>VIQ</th>\n",
       "      <th>PIQ</th>\n",
       "      <th>Weight</th>\n",
       "      <th>Height</th>\n",
       "      <th>MRI_Count</th>\n",
       "      <th>partY</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Female</td>\n",
       "      <td>133</td>\n",
       "      <td>132</td>\n",
       "      <td>124</td>\n",
       "      <td>118.0</td>\n",
       "      <td>64.5</td>\n",
       "      <td>816932</td>\n",
       "      <td>-0.357519</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Male</td>\n",
       "      <td>140</td>\n",
       "      <td>150</td>\n",
       "      <td>124</td>\n",
       "      <td>NaN</td>\n",
       "      <td>72.5</td>\n",
       "      <td>1001121</td>\n",
       "      <td>0.148448</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Male</td>\n",
       "      <td>139</td>\n",
       "      <td>123</td>\n",
       "      <td>150</td>\n",
       "      <td>143.0</td>\n",
       "      <td>73.3</td>\n",
       "      <td>1038437</td>\n",
       "      <td>0.993531</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Male</td>\n",
       "      <td>133</td>\n",
       "      <td>129</td>\n",
       "      <td>128</td>\n",
       "      <td>172.0</td>\n",
       "      <td>68.8</td>\n",
       "      <td>965353</td>\n",
       "      <td>1.838968</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Female</td>\n",
       "      <td>137</td>\n",
       "      <td>132</td>\n",
       "      <td>134</td>\n",
       "      <td>147.0</td>\n",
       "      <td>65.0</td>\n",
       "      <td>951545</td>\n",
       "      <td>-0.744026</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Gender  FSIQ  VIQ  PIQ  Weight  Height  MRI_Count     partY\n",
       "1  Female   133  132  124   118.0    64.5     816932 -0.357519\n",
       "2    Male   140  150  124     NaN    72.5    1001121  0.148448\n",
       "3    Male   139  123  150   143.0    73.3    1038437  0.993531\n",
       "4    Male   133  129  128   172.0    68.8     965353  1.838968\n",
       "5  Female   137  132  134   147.0    65.0     951545 -0.744026"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "brain.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x7f91b4b2d110>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot partY var as a function of FSIQ and VIQ\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.scatter(x = brain[\"PIQ\"], y = brain[\"partY\"], s = brain[\"FSIQ\"]*4, alpha = 0.5, label = \"FSIQ\")\n",
    "plt.title(\"partY as a function of PIQ and FSIQ\")\n",
    "plt.xlabel(\"PIQ\")\n",
    "plt.ylabel(\"partY\")\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The partY variable seems to increase with FSIQ and PIQ."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:                  partY   R-squared:                       0.170\n",
      "Model:                            OLS   Adj. R-squared:                  0.148\n",
      "Method:                 Least Squares   F-statistic:                     7.764\n",
      "Date:                Fri, 29 May 2020   Prob (F-statistic):            0.00827\n",
      "Time:                        17:08:57   Log-Likelihood:                -50.962\n",
      "No. Observations:                  40   AIC:                             105.9\n",
      "Df Residuals:                      38   BIC:                             109.3\n",
      "Df Model:                           1                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "Intercept     -0.8531      0.392     -2.174      0.036      -1.648      -0.059\n",
      "FSIQ:PIQ    7.803e-05    2.8e-05      2.786      0.008    2.13e-05       0.000\n",
      "==============================================================================\n",
      "Omnibus:                        1.332   Durbin-Watson:                   2.090\n",
      "Prob(Omnibus):                  0.514   Jarque-Bera (JB):                1.202\n",
      "Skew:                           0.402   Prob(JB):                        0.548\n",
      "Kurtosis:                       2.729   Cond. No.                     3.92e+04\n",
      "==============================================================================\n",
      "\n",
      "Warnings:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 3.92e+04. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    }
   ],
   "source": [
    "# create linear model with interaction only\n",
    "from statsmodels.formula.api import ols\n",
    "model1 = ols(\"partY ~ FSIQ:PIQ\", brain).fit()\n",
    "print(model1.summary())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The interaction between FSIQ and PIQ was a significant predictor of partY (*b*=0.00007, *t*=2.79, *p*=.008). The overall model significantly predicted partY (F(1,38)=7.76, *p*=.008, $R^{2}$=.17)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "np.random.seed(312)\n",
    "#crete second random variable\n",
    "partY2 = np.random.randn(len(brain))\n",
    "\n",
    "#add new var to data\n",
    "brain[\"partY2\"] = partY2\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:                 partY2   R-squared:                       0.027\n",
      "Model:                            OLS   Adj. R-squared:                  0.001\n",
      "Method:                 Least Squares   F-statistic:                     1.037\n",
      "Date:                Fri, 29 May 2020   Prob (F-statistic):              0.315\n",
      "Time:                        17:08:57   Log-Likelihood:                -55.477\n",
      "No. Observations:                  40   AIC:                             115.0\n",
      "Df Residuals:                      38   BIC:                             118.3\n",
      "Df Model:                           1                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==============================================================================\n",
      "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------\n",
      "Intercept      0.3261      0.439      0.742      0.463      -0.563       1.216\n",
      "FSIQ:PIQ   -3.192e-05   3.14e-05     -1.018      0.315   -9.54e-05    3.15e-05\n",
      "==============================================================================\n",
      "Omnibus:                        0.360   Durbin-Watson:                   2.108\n",
      "Prob(Omnibus):                  0.835   Jarque-Bera (JB):                0.491\n",
      "Skew:                          -0.192   Prob(JB):                        0.783\n",
      "Kurtosis:                       2.618   Cond. No.                     3.92e+04\n",
      "==============================================================================\n",
      "\n",
      "Warnings:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The condition number is large, 3.92e+04. This might indicate that there are\n",
      "strong multicollinearity or other numerical problems.\n"
     ]
    }
   ],
   "source": [
    "model2 = ols(\"partY2 ~ FSIQ:PIQ\", brain).fit()\n",
    "print(model2.summary())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "scrolled": false
   },
   "source": [
    "The interaction of FSIQ and PIQ did not significantly predict partY2 (F(1,38)=1.01, *p*=.32, $R^{2}$=.03)."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}