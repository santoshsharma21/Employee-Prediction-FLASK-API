import joblib
import pandas as pd
from catboost import CatBoostClassifier

# set parameters
params = {}
params['iterations'] = 823
params['depth'] = 7
params['learning_rate'] = 0.010885582797311217
params['random_strength'] = 0.6024649278875751
params['l2_leaf_reg'] = 0.89274890886717
params['bagging_temperature'] = 0.9323437952247747
params['scale_pos_weight'] = 3
params['verbose'] = 0
params['random_state'] = 21

model = CatBoostClassifier(**params)


if __name__ == "__main__":
    # Read data
    xtrain = pd.read_csv('../input/processed_train.csv')
    xtest = pd.read_csv('../input/processed_test.csv')
    sub = pd.read_csv('../input/sample_submission.csv')
    
    # predictors, target
    ytrain = xtrain['is_promoted'].copy()
    xtrain.drop(['is_promoted'], axis=1, inplace=True)
    # print shape of data
    print(f"Train shape = {xtrain.shape}, Test shape = {xtest.shape}")
    
    # Train model
    print('Training Model........')
    model.fit(xtrain, ytrain)
    test_pred = model.predict(xtest)
    #print(f"features - {model.feature_names_}")

    # create submission file
    print('Writing submission file......')
    sub['is_promoted'] = test_pred
    sub.to_csv('../input/clf_catboost_sub_file.csv', index = False)

    # save model
    print('Saving trained model.......')
    joblib.dump(model,'../model/ml_clf')

    print('Model training and saving completed......')