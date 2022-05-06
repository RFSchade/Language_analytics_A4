#======================================================#
#=============> Word embedding CNN Model <=============#
#======================================================#

#=====> Import modules
# simple text processing tools
import re
import tqdm
import os
import contractions
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')

# data wranling
import pandas as pd
import numpy as np

# tensorflow
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, 
                                    Flatten,
                                    Conv1D, 
                                    MaxPooling1D, 
                                    Embedding)
from tensorflow.keras.preprocessing.text import Tokenizer ##
from tensorflow.keras.preprocessing import sequence

# scikit-learn
from sklearn.metrics import (confusion_matrix, 
                            classification_report)
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# visualisations 
import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

#=====> Define global variabes
# Max sequence length
MX_SEQUENCE_LENGTH = 1000 
# Number of dimentions for embeddings 
EMBED_SIZE = 300
# Number of epochs
EPOCHS = 2
# Batch size
BATCH_SIZE = 128

#=====> Define functions
# > Load data 
def prep_data():
    # Print info 
    print("[info] Loading data...")
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
                                                        test_size=0.2, # 80-20% split
                                                        random_state=42) # Set seed
    # Print info
    print("[info] Data loaded")
    print("[info] Training data = 80%, Test data = 20%")
    
    return (X_train, X_test, y_train, y_test)

# > Tokenize text
def tokenize(X_train, X_test):
    # Print info
    print("[INFO] Tokenizing text...")
    
    # Define out of vocabulary token
    t = Tokenizer(oov_token = "<UNK>") 
    # Fit the tokenizer on the documents 
    t.fit_on_texts(X_train)
    # Set padding value (number the data is patted with?)
    t.word_index["<PAD>"] = 0
    
    # Converting data to sequences
    X_train_seqs = t.texts_to_sequences(X_train)
    X_test_seqs = t.texts_to_sequences(X_test)
    
    # Defining vocabulary size 
    # (This is capitlized, so it is supposed to be a global variable) 
    # (But it needs the object "t" for defenition)
    # (should I decapitalize it? or can I somwhow still define it in the beggining?)
    VOCAB_SIZE = len(t.word_index)
    
    # Print info 
    print(f"[INFO] Vocabulary size = {len(t.word_index)}")
    print(f"[INFO] Number of documents/sequences = {t.document_count}")
    
    return (X_train_seqs, X_test_seqs, VOCAB_SIZE)

# > Normalize sequences
def normalize(X_train_seqs, X_test_seqs, y_train, y_test):
    # Print info 
    print("[INFO] Normalizing data...")
    
    # Padding sequences
    X_train_pad = sequence.pad_sequences(X_train_seqs, maxlen = MX_SEQUENCE_LENGTH) 
    # Can change padding using  padding = "post"
    X_test_pad = sequence.pad_sequences(X_test_seqs, maxlen = MX_SEQUENCE_LENGTH)
    
    # Create one-hot encodings (Still not sure what this does)
    lb = LabelBinarizer()
    y_train_lb = lb.fit_transform(y_train)
    y_test_lb = lb.fit_transform(y_test)
    
    return (X_train_pad, X_test_pad, y_train_lb, y_test_lb)

# > Create model
def build_model(VOCAB_SIZE, EMBED_SIZE, MX_SEQUENCE_LENGTH):
    # Print info
    print("[INFO] Building model...")
    
    # create the model
    model = Sequential()
    # embedding layer
    model.add(Embedding(VOCAB_SIZE, 
                        EMBED_SIZE, 
                        input_length=MX_SEQUENCE_LENGTH))

    # first convolution layer and pooling
    model.add(Conv1D(filters=128, 
                            kernel_size=4, 
                            padding='same',
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # second convolution layer and pooling
    model.add(Conv1D(filters=64, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # Third Convolutional layer and pooling 
    model.add(Conv1D(filters=32, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # fully-connected classification layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                            optimizer='adam', 
                            metrics=['accuracy'])
    # print info
    print("[INFO] Model summary:")
    model.summary()
    
    return model
    
# > Evaluate model
# > Evaluate model
def evaluate(model, X_test_pad, y_test_lb):
    # evaluate nertwork with 0.5 decision boundary
    predictions = (model.predict(X_test_pad) > 0.5).astype("int32")
    # assign labels
    predictions = [0 if item == 1 else 1 for item in predictions]
    
    # Initialize label names
    labels = ["Nontoxic", "Toxic"]
    # create classification report
    report = classification_report(y_test_lb, 
                                   predictions, 
                                   target_names=labels)
    # Print metrics
    print(report)
    
    # Save metrics
    outpath = os.path.join("output", "CNN_report.txt")
    with open(outpath, "w") as f:
        f.write(report)

#=====> Define main()
def main():
    (X_train, X_test, y_train, y_test) = prep_data()
    (X_train_seqs, X_test_seqs, VOCAB_SIZE) = tokenize(X_train, X_test)
    (X_train_pad, X_test_pad, y_train_lb, y_test_lb) = normalize(X_train_seqs, X_test_seqs, y_train, y_test)
    model = build_model(VOCAB_SIZE, EMBED_SIZE, MX_SEQUENCE_LENGTH)
    
    # Fitting
    history = model.fit(X_train_pad, y_train,
                       epochs = EPOCHS,
                       batch_size = BATCH_SIZE,
                       validation_split = 0.1, # second validation split  
                       verbose = True)
    # Reporting
    evaluate(model, X_test_pad, y_test_lb)

# Run main() function from terminal only
if __name__ == "__main__":
    main()
