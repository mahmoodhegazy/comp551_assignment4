import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report

def main():
    FILENAME = "data/trial1/train_input.csv"
    CATEGORY = "data/trial1/train_output.csv"
    TEST_SET = "data/test_input.csv"
    OUTPUT = "output/predictions.csv"
    
    train_input_X = pd.read_csv(FILENAME, usecols=["conversation"])
    train_input_Y = pd.read_csv(CATEGORY, usecols=["category"])
    train_input_XY = pd.concat([train_input_X, train_input_Y], axis=1)

    test_input = pd.read_csv(TEST_SET, usecols=["conversation"])

    predictions = naive_bayes(train_input_XY, test_input)
    test_input["category"] = predictions
    test_input.to_csv("output/predictions-bayes.csv")

    predictions = svm(train_input_XY, test_input)
    test_input["category"] = predictions
    test_input.to_csv("output/predictions-svm.csv")

    predictions = decision_tree(train_input_XY, test_input)
    test_input["category"] = predictions
    test_input.to_csv("output/predictions-decisiontrees.csv")


def run_pipeline(pipeline, data, testSet):
    predictionSet = []
    
    for train, test in KFold(n=len(data), n_folds=3):
        conversationsX = data.iloc[train]["conversation"].values
        conversationsY = data.iloc[train]["category"].values

        testX = data.iloc[test]["conversation"].values
        testY = data.iloc[test]["category"].values

        predictX = testSet["conversation"].values

        pipeline.fit(conversationsX, conversationsY)
        testYResults = pipeline.predict(testX)

        report = classification_report(testY, testYResults)
        print report

        predictions = pipeline.predict(predictX)
        predictionSet.append(predictions)

    return predictionSet

def naive_bayes(data, testSet):
    print "Run Naive Bayes"
    pipeline = Pipeline([
        ('count',  CountVectorizer(ngram_range=(1, 2))),
        ('tfidf',  TfidfTransformer()),
        ('classify',  MultinomialNB())
    ])
    
    print "Splitting into training and testing"
    cutoff = np.random.rand(len(data)) < 0.7
    train = data[cutoff]
    test = data[~cutoff]

    conversationsX = train["conversation"].values
    conversationsY = train["category"].values

    testX = test["conversation"].values
    testY = test["category"].values

    predictX = testSet["conversation"].values

    pipeline.fit(conversationsX, conversationsY)
    testYResults = pipeline.predict(testX)

    report = classification_report(testY, testYResults)
    print report

    predictions = pipeline.predict(predictX)
    return predictions

def svm():
    print "Run SVM"
    pipeline = Pipeline([
        ('count',  CountVectorizer()),
        ('tfidf',  TfidfTransformer()),
        ('classify',  SVC(kernel='rbf'))
    ])
    
    print "Splitting into training and testing"
    cutoff = np.random.rand(len(data)) < 0.7
    train = data[cutoff]
    test = data[~cutoff]

    conversationsX = train["conversation"].values
    conversationsY = train["category"].values

    testX = test["conversation"].values
    testY = test["category"].values

    predictX = testSet["conversation"].values

    pipeline.fit(conversationsX, conversationsY)
    testYResults = pipeline.predict(testX)

    report = classification_report(testY, testYResults)
    print report

    predictions = pipeline.predict(predictX)
    return predictions

def decision_tree():
    print "Run Decision Tree"
    pipeline = Pipeline([
        ('count',  CountVectorizer(ngram_range=(1, 2))),
        ('tfidf',  TfidfTransformer()),
        ('classify',  DecisionTreeClassifier())
    ])

    print "Splitting into training and testing"
    cutoff = np.random.rand(len(data)) < 0.7
    train = data[cutoff]
    test = data[~cutoff]

    conversationsX = train["conversation"].values
    conversationsY = train["category"].values

    testX = test["conversation"].values
    testY = test["category"].values

    predictX = testSet["conversation"].values

    pipeline.fit(conversationsX, conversationsY)
    testYResults = pipeline.predict(testX)

    report = classification_report(testY, testYResults)
    print report

    predictions = pipeline.predict(predictX)
    return predictions

if __name__ == "__main__":
    main()