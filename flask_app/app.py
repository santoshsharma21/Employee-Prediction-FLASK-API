# import libraries
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request


# load trained model
model = joblib.load('ml_clf')
avg_training_by_dept = joblib.load('fea_training_avg')
avg_rating_by_dept = joblib.load('fea_rating_avg')
encoder = joblib.load('cat_encoder')


app = Flask(__name__)
df = pd.DataFrame()

@app.route("/")
def home():
    return render_template('promotion.html')

@app.route('/predict', methods = ['POST'])
def predict():
    keys = [i for i in request.form.keys()]
    vals = [i for i in request.form.values()]
    dic = dict(zip(keys,vals))
    df = pd.DataFrame(dic, index = [0])
    # create new features
    df['avg_training_score_by_dept'] = df['department'].map(avg_training_by_dept)
    df['avg_rating_by_dept'] = df['department'].map(avg_rating_by_dept)
    # ordinal encoding
    col = ['department']
    df[col] = encoder.transform(df[col])
    # get prediction
    pred_prob = np.round(model.predict_proba(df.values)[:,1][0],2)*100
    return render_template('promotion.html', prediction_text = f'Probability of getting Promotion is : {pred_prob}%')


if __name__ == "__main__":
    app.run()