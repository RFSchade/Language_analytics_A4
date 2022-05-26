#============================================================#
#=============> Logistic Regression Classifier <=============#
#============================================================#

#=====> Import modules
# System tools
import os
import sys
sys.path.append(os.getcwd())
import argparse

# Import teaching utils
import numpy as np
import pandas as pd

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

#=====> Define Functions
# > Load and prepare data 
def prep_data(split):
    # Print info 
    print("[INFO] Loading data...")
    # Load data 
    filepath = os.path.join("in", "VideoCommentsThreatCorpus.csv")
    data = pd.read_csv(filepath)
    # Balancing data using some code I cribbed from the utils.balance() function
    data_balance = (data.groupby('label', as_index=False)
                        .apply(lambda x: x.sample(n=1000))
                        .reset_index(drop=True))
    # Splitting X and y 
    X = data_balance["text"]
    y = data_balance["label"]
    # Splitting data 
    X_train, X_test, y_train, y_test = train_test_split(X, # data
                                                        y, # labels
                                                        test_size=split, # test/training split
                                                        random_state=42) # Set seed
    # Print info
    print("[INFO] Data loaded")
    
    return (X_train, X_test, y_train, y_test)

# > Vectorize data
def vectorize(X_train, X_test, max_df, min_df, max_features):
    # Print info 
    print("[INFO] Vectorizing data...")
    vectorizer = TfidfVectorizer(ngram_range = (1,2), # unigrams and bygrams (no trigrams)
                                 lowercase = True, # Make words lowercase
                                 max_df = max_df, # remove very common words (df = documnet frequency)
                                 min_df = min_df, # remove very rare words
                                 max_features = max_features) # keep features limited (limits the nr of words) 
                                                     # limitig features keeps the model for overfitting in small data
    # Fit to the training data 
    X_train_feats = vectorizer.fit_transform(X_train)
    # Then we transform test data to match training data 
    X_test_feats = vectorizer.transform(X_test)
    # Print info
    print("[INFO] Data vectorized")
    
    return (X_train_feats, X_test_feats)

# > Classify and report
def classify(X_train_feats, y_train, X_test_feats, y_test):
    # Print info
    print("[INFO] Creating classifier")
    # Create classifier
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)
    # Make predictions
    y_pred = classifier.predict(X_test_feats)
    # Initialize label names
    labels = ["Nontoxic", "Toxic"]
    # Get classifer metrics 
    classifier_metrics = metrics.classification_report(y_test, y_pred, target_names=labels)
    # Print info
    print("[info] Printing classification report:")
    print(classifier_metrics)
    # Save metrics
    outpath = os.path.join("output", "logistic_report.txt")
    with open(outpath, "w") as f:
        f.write(classifier_metrics)
    # Print info 
    print("[INFO] Classification report can be found as a .txt file in the 'outpot' directory")
    
# > Parse arguments
def parse_args(): 
    # Initialize argparse
    ap = argparse.ArgumentParser()
    # Commandline parameters 
    ap.add_argument("-s", "--split", 
                    required=False, 
                    type=float,
                    help="Proportion of the data that goes into the test dataset - default is 0.2", 
                    default=0.2)
    ap.add_argument("-max", "--max_df", 
                    required=False, 
                    type=float,
                    help="Threshold for removal of the most common words in sequence - default is 0.95", 
                    default= 0.95)
    ap.add_argument("-min", "--min_df", 
                    required=False, 
                    type=float,
                    help="Threshold for removal of the least common words in sequence - default is 0.05", 
                    default= 0.05)
    ap.add_argument("-f", "--max_features", 
                    required=False, 
                    type=int,
                    help="Max nr. of features after vectorizing - default is 500", 
                    default= 500)
    # Parse argument
    args = vars(ap.parse_args())
    # return list of argumnets 
    return args

#=====> Define main()
def main():
    # Get argument
    args = parse_args()
    
    # Loading and preparing data
    X_train, X_test, y_train, y_test = prep_data(args["split"])
    # Vectorizing data
    X_train_feats, X_test_feats = vectorize(X_train, X_test, args["max_df"], args["min_df"], args["max_features"])
    # Training model and reporting classification metrics 
    classify(X_train_feats, y_train, X_test_feats, y_test)
    
    # Print info 
    print("[INFO] Job complete")

# Run main() function from terminal only
if __name__ == "__main__":
    main()