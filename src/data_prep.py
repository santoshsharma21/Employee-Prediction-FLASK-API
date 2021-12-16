import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# useful functions
def GetAvgVar(groupby_df, mapto_df, groupby_var, compute_var):
    avg_dic = groupby_df.groupby(groupby_var)[compute_var].mean().to_dict()
    value = mapto_df[groupby_var].map(avg_dic)
    return value, avg_dic

if __name__ == "__main__":
    # Read data
    xtrain = pd.read_csv('../input/train.csv')
    xtest = pd.read_csv('../input/test.csv')
    target = xtrain['is_promoted'].values
    xtrain.drop('is_promoted', axis = 1, inplace =True)
    # correct columns name
    df_cols = {
               'KPIs_met >80%' : 'kpi',
               'awards_won?' : 'awards_won'
              }
    xtrain.rename(columns = df_cols, inplace =True)
    xtest.rename(columns = df_cols, inplace =True)

    # print shape of data
    print(f"Train shape = {xtrain.shape}, Test shape = {xtest.shape}")

    # Impute mising value
    xtrain['previous_year_rating'] = xtrain['previous_year_rating'].fillna(0)
    xtest['previous_year_rating'] = xtest['previous_year_rating'].fillna(0)
    
    # creating new features
    # train
    xtrain['avg_training_score_by_dept'],training_avg = GetAvgVar(xtrain, xtrain, 'department', 'avg_training_score')
    xtrain['avg_rating_by_dept'],rating_avg = GetAvgVar(xtrain, xtrain, 'department', 'previous_year_rating')
    joblib.dump(training_avg, '../model/fea_training_avg')
    joblib.dump(rating_avg, '../model/fea_rating_avg')

    # test
    xtest['avg_training_score_by_dept'],_ = GetAvgVar(xtrain, xtest, 'department', 'avg_training_score')
    xtest['avg_rating_by_dept'],_ = GetAvgVar(xtrain, xtest, 'department', 'previous_year_rating')
    
    # cat feature encoding
    cat_cols = ['department']
    oe = OrdinalEncoder()
    xtrain[cat_cols] = oe.fit_transform(xtrain[cat_cols])
    xtest[cat_cols] = oe.transform(xtest[cat_cols])
    # dump ordinal encoder
    joblib.dump(oe, '../model/cat_encoder')

    # drop unused columns
    unused_cols = ['employee_id','education','gender','region','recruitment_channel']
    xtrain.drop(unused_cols, axis=1, inplace=True)
    xtest.drop(unused_cols, axis=1, inplace=True)

    # write csv files
    xtrain['is_promoted'] = target
    xtrain.to_csv('../input/processed_train.csv', index = False)
    xtest.to_csv('../input/processed_test.csv', index = False)

    print('data prep complete..........')