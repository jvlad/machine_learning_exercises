## Thu, Jul 20, 11:45:01 PM 20230721-0445

[Questions
  [Picking performance evaluation metric for regression model. 
    When you use a regression model, do you follow any formal approach for picking a performance evaluation metric (or a combination of them)?  

    ...MAE/MAPE, MSE, RMSE, R-squared, Adjusted R-squared – I've read through a number of overviews, all are "the best". However, I started wondering if choosing it is a matter of consensus like "use whatever my fellows used".

    https://www.linkedin.com/posts/vladzams_datascience-activity-7129966707267301376-XQ_E?utm_source=share&utm_medium=member_desktop]

  [Do we need to scale target variable?  
    opinions are controversial, need to explore further  
  ]
]

[Apriori algorithm
  Generally there is no point in selecting a base product that has a low support itself (e. g. 5-10%). Yet, technically there is no problem in doing so if we have computation resources.  

  [Algorithm
    sort all the found rules by the lift DESC  
    The higher the lift, the better chances that recommending i2 when there was i1 selected by user, will lead users' attention to i2  
  ]
  
  [support(i) = len(transactions with product i) / count(all items)]
  [confidence(i1 -> i2) = len(transactions with both product i1 and i2) / len(transactions with product i1)]
  [lift (i1 -> i2) = confidence(i1 -> i2) / support (i2), 
    where i2 is the item that interest us, and i1 – the base item]
]

[Clustering
  [Hierarchical
    [Ward method – a way to minimize the variance within the clusters. Was used together with hierarchical clustering
    ]

    [2 types exist: Divisive and Agglomerative]

    [Pros: The optimal number of clusters can be obtained by the model itself, practical visualisation with the dendrogram
      
      Cons: Not appropriate for large datasets
    ]

    ['SciPy Hierarchical Clustering and Dendrogram Tutorial'  
      Thu Dec 14 01:01:59 EST 2023  
      https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    ]
  ]

  [K-Means
    Pros: Simple to understand, easily adaptable, works well on small or large datasets, fast, efficient and performant
    Const: Need to choose the number of clusters (elbow method)
  ]
]

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


[Building a multiple linear regression model  
  We need to pick which independent variables out of those that we have we're going to take into account
  
  [There are 5 methods for this:  
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
]

[[] Prediction error types
  False positive - Type 1 error
  False negative - Type 2 error
]

[[] Ensemble learning – a compound method – a one that combines few other ML methods. Or, sometimes, the same method applied multiple times (e. g. random forest)
]

[[] Random Forest
  1. Pick at random K data points from the Training set.
  2. Build the Decision Tree associated to these K data points.
  3. Choose the number Ntree of trees you want to build and repeat STEPS 1 & 2
  4. For a new data point, make each one of your Ntree trees predict the category to which the data points belongs, and assign the new data point to the category that wins the majority vote.
]

[[] Evaluating performance of classification models
  accuracy ratio should not be used as the only indicator due to possible Accuracy Paradox. CAP (Cumulative Accuracy Profile)
]

