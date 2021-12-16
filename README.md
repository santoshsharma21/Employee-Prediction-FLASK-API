# EMPLOYEE PROMOTION PREDICTION
## DATA :-
Data was taken from Analytics Vidhya.

Link :- https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018-1/

## PROBLEM STATEMENT :-
To identify eligible candidates for promotion based on various attributes around employee's past and current performance.


## APPROACH USED :-
* Data visualization : Used Matplotlib to visualize distribution of data.
* Data transformation : Used Ordinal Encoder to convert categorical data into numeric data.
* Feature Engineering : Created new features.
* Hyperparameter tuning : Used optuna to find best set of parameters.
* Model Selection : Model selected based on cv-score, as well as public LB-sccore.
* Deployment : Deployed on heroku.

## USER INTERFACE :-
Link :- https://employee-promotion.herokuapp.com/

![pic_1](screenshot/pic_2.png)

## TECHNOLOGY USED :-
* Python
* Scikit-learn 
* Catboost
* Optuna
* Flask
* HTML

