from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
import csv
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import pandas


def perform(classifiers, vectorizers, train_data, test_data):
    for classifier in classifiers:
        for vectorizer in vectorizers:
            string = ''
            string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

            # train
            vectorize_text = vectorizer.fit_transform(train_data.v2)
            classifier.fit(vectorize_text, train_data.v1)

            # score
            vectorize_text = vectorizer.transform(test_data.v2)
            score = classifier.score(vectorize_text, test_data.v1)
            string += '. Has score: ' + str(score)
            print(string)


# open data set and divide it
data = pandas.read_csv('/Users/ilyasjaghoori/Downloads/spam.csv', encoding='latin-1')
learn = data[:4400]  # 4400 items
test = data[4400:]  # 1172 items

perform(
    [
        BernoulliNB(),
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        AdaBoostClassifier(),
        BaggingClassifier(),
        ExtraTreesClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier(),
        CalibratedClassifierCV(),
        DummyClassifier(),
        PassiveAggressiveClassifier(),
        RidgeClassifier(),
        RidgeClassifierCV(),
        SGDClassifier(),
        OneVsRestClassifier(SVC(kernel='linear')),
        OneVsRestClassifier(LogisticRegression()),
        KNeighborsClassifier()
    ],
    [
        CountVectorizer(),
        TfidfVectorizer(),
        HashingVectorizer()
    ],

    learn,
    test
)

data2 = pandas.read_csv('/Users/ilyasjaghoori/Downloads/spam.csv', encoding='latin-1')
train_data = data2[:4400]
test_data = data[4400:]

classifier = OneVsRestClassifier(SVC(kernel='linear'))
vectorizer = TfidfVectorizer()

# train
vectorize_text = vectorizer.fit_transform(train_data.v2)
classifier.fit(vectorize_text, train_data.v1)

csv_arr = []
for index, row in test_data.iterrows():
    answer = row[0]
    text = row[1]
    vectorize_text = vectorizer.transform([text])
    predict = classifier.predict(vectorize_text)[0]
    if predict == answer:
        result = 'right'

    else:
        result = 'wrong'
    csv_arr.append([len(csv_arr), text, answer, predict, result])

# write csv
with open('test_score.csv', 'w', newline='') as csvfile:
    spamwriter =  csv.writer(csvfile, delimiter=';',
                             quotechar = '"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['#', 'text', 'answer', 'predict', result])

    for row in csv_arr:
        spamwriter.writerow(row)



app = Flask(__name__)
global Classifier
global Vectorizer

# load data
data3 = pandas.read_csv('/Users/ilyasjaghoori/Downloads/spam.csv', encoding='latin-1')
train_data = data[:4400]
test_data = data[4400:]

# train model
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier.fit(vectorize_text, train_data.v1)

@app.route('/', methods=['GET'])
def index():
    message = request.args.get('message', 'CLICK THIS LINK IMMEDIATELY FOR MONEY')
    error = ''
    predict_proba = ''
    predict = ''

    global Classifier
    global Vectorizer
    try:
        if len(message) > 0:
            vectorize_message = Vectorizer.transform([message])
            predict = Classifier.predict(vectorize_message)[0]
            predict_proba = Classifier.predict_proba(vectorize_message).tolist()
    except BaseException as inst:
        error = str(type(inst).__name__) + ' ' + str(inst)
    return jsonify(
        message = message, predict_proba = predict_proba,
        predict=predict, error=error
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)



