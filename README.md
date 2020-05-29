# Linear Regression with Statsmodels

This repo contains an anlalysis of brainsize and intelligence. All of the relevant files can be found in the `practical` folder. 
The data used in this anlaysis can be found in `brainsize.csv`
The code and anlysis can be found in `myanlysis.ipynb` [here](https://github.com/Alex-A14/Albury-A-QLSC612/blob/master/practical/myanalysis.ipynb)
Or you can view an html [here](https://htmlpreview.github.io/?https://github.com/Alex-A14/Albury-A-QLSC612/blob/master/practical/myanalysis.html)

A description of the full build environment is provided in `requirements.txt` However, the analysis primarily relies on `numpy` `pandas` `matplotlib` and `statsmodels`

## Overview

The `brainsize` data includes the demographic variables height, weight, and gender, as well as measures of intelligence including Verbal IQ (VIQ), Performance IQ (PIQ), and Full Scale IQ (FSIQ). These measures were obtained using the Wechsler Intelligence Scale for Children (WISC).

The analysis aims to predict an observed variable partY using the data collected in the brainsize dataset.

## Method

The primary analysis in this repo is ols regression using `statsmodels`

The partY variable was drawn from a random normal distribution.

## Results

The analysis found that the best predictor of the partY variable was an interaction between PIQ and FSIQ. It is important to note that the model included only the interaction beetween PIQ and FSIQ, and not their main effects. Other combinations of predictors showed no improvement in model fit, including the addition of polynomials.

The final form of the model is as follows:

`partY ~ FSIQ:PIQ`

A second random variable, `partY2`, was analyzed with the same predictors, and no significant relationship was found.