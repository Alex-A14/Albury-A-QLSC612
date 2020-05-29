# Linear Regression with Statsmodels

This repo contains an anlalysis of brainsize and intelligence. All of the relevant files can be found in the `practical` folder. 
The data used in this anlaysis can be found in `brainsize.csv`
The code and output can be found in `myanlysis.ipynb`

A full build environment is provided in `requirements.txt` However, the analysis primarily relies on `numpy` `pandas` `matplotlib` and `statsmodels`

## Overview

The `brainsize` data includes the demographic variables height, weight, and gender, as well as measures of intelligence including Verbal IQ (VIQ), Performance IQ (PIQ), and Full Scale IQ (FSIQ). These measures were obtained using the Wechsler Intelligence Scale for Children (WISC).

The analysis aims to predict an observed variable partY using the data collected in the brainsize dataset.

## Method

The primary analysis in this repo is ols regression using `statsmodels`

The partY variable was 

## Results

The analysis found the the best predictor of the partY variable was an interaction between PIQ and FSIQ. It is important to note that the model included only the interaction beetween PIQ and FSIQ, and not their main effects.