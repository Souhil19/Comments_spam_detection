import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    files=['./youtube-dataset/Youtube04.csv',
    './youtube-dataset/Youtube01.csv',
    './youtube-dataset/Youtube02.csv',
    './youtube-dataset/Youtube05.csv',
    './youtube-dataset/Youtube03.csv']

    all_df=[]
    for i in files:
        all_df.append(pd.read_csv(i).drop(['COMMENT_ID', 'AUTHOR', 'DATE'], axis=1))

    data = pd.concat(all_df, axis=0, ignore_index=True)

    inputs = data['CONTENT']
    target = data['CLASS']

    x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2,
                                                        random_state=365,
                                                        stratify=target)

    vectorizer = CountVectorizer()
    x_train_transf = vectorizer.fit_transform(x_train)
    x_test_transf = vectorizer.transform(x_test)

    x_train_transf.toarray()
    clf = MultinomialNB(class_prior=np.array([0.6, 0.4]))
    clf.fit(x_train_transf, y_train)
    np.exp(clf.class_log_prior_)


    comment = request.form['comment']
    predict_data = vectorizer.transform([comment])

    prediction = model.predict(predict_data)
    return render_template("index.html", prediction_text = prediction, name=comment.upper())



if __name__ == '__main__':
    app.run()
