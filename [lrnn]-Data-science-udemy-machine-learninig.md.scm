## Thu, Jul 20, 11:45:01 PM 20230721-0445

## Statistical Models 
Logistic regression

Resources:  
https://drive.google.com/drive/folders/1OFNnrHRZPZ3unWdErjLHod8Ibv2FfG1d  
https://drive.google.com/drive/folders/16s9Obk1t_4Iaa2EBzgh0MQADOK6ReCst
https://www.superdatascience.com/pages/machine-learning

382 – total lectures

## General Setup

import libs:  
    numpy
    matplotlib.pyplot
    pandas  
    sklearn

scikit-learn - lib for data transformation

## Statistics Course (Krish Naik)
https://youtu.be/LZzq1zSL1bs?t=3865


[Code Samples
  in [ML_Toolkit_Reference.ipynb]
]

[Building a multiple linear regression model  
  We need to pick which independent variables out of those that we have we're going to take into account
  There are 5 methods for this  
  1. all-in (not recommended unless it's a preparation step for Backward Elimination)

  2. Backward Elimination (an instance of Stepwise regression)
    one-by-one exclude all the variables which P-value is bigger than arbitrary picked significance level. E. g. SL = 0.05. Excluding variables with a high P-value means that we excluding the variables changing which doesn't affect the system. I. e. the observed system behavior is highly probably to occur regardless of those variables. 

  3. Forward selection (an instance of Stepwise regression)
    Building simple linear regressions using separately each of the independent variables. Out of those n [simple linear regression] models, choose one model – x_i model – with the lowest P-value (where P should also be lower than SL).  

    Keep x_n and check all possible models with one extra variable. Again, build the rest possible (n-1) models and pick the one that has the lowest P-value (and smaller than SL)

    As soon as there are no more variables with P < SL, we're done. So, our model will include not all the n of x, but only a subset.  

  4. Bidirectional Elimination (an instance of [Stepwise regression] sometimes referred to as [Stepwise regression] itself) – combines [Forward selection] and [Backward Elimination]  
    Perform one step of the [Forward selection]
    Perform all the steps of [Backward Elimination]  

    1. Select [SL-to-enter, SL-to-stay]  
    2. do Step#1 of the [forward selection] method: pick x_i with the lowest P-value which should also be less than SL-to-enter  
    3. do all the steps of the [backward elimination] – try to eliminate as many variables as possible using SL-to-stay
    4. repeat step#2: pick x_j with the lowest P-value which should also be less than SL-to-enter  
    Eventually we will end up with a state where we can't add more variables nor we can eliminate more variables. That would be our selected set of variables for the model  

  5. Score Comparison (all possible models)
  Construct all possible models, pick the one with the best criterion  
]