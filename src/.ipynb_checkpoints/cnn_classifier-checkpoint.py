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

# User-experience
import argparse

# visualisations 
import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

#=====> Define global variabes
# Max sequence length
MX_SEQUENCE_LENGTH = 500
# Number of dimentions for embeddings 
EMBED_SIZE = 300
# Batch size
BATCH_SIZE = 128

#=====> Define functions
# > Load data 
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
                                                        test_size=split,
                                                        random_state=42) # Set seed
    # Print info
    print("[INFO] Data loaded")
    
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
    vocab_size = len(t.word_index)
    
    # Print info 
    print(f"[INFO] Vocabulary size = {vocab_size}")
    print(f"[INFO] Number of documents/sequences = {t.document_count}")
    
    return (X_train_seqs, X_test_seqs, vocab_size)

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

# > Plot history
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    # Saving image
    plt.savefig(os.path.join("output", "history_img.png"))

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
    ap.add_argument("-e", "--epochs", 
                    required=False, 
                    type=int,
                    help="Nr. of epochs for the CNN to run - default is 10", 
                    default= 10)
    # Parse argument
    args = vars(ap.parse_args())
    # return list of argumnets 
    return args

#=====> Define main()
def main():
    # Get argument
    args = parse_args()
    
    # Loading and preparing data
    (X_train, X_test, y_train, y_test) = prep_data(args["split"])
    # Tokenize data
    (X_train_seqs, X_test_seqs, vocab_size) = tokenize(X_train, X_test)
    # Normalize data 
    (X_train_pad, X_test_pad, y_train_lb, y_test_lb) = normalize(X_train_seqs, X_test_seqs, y_train, y_test)
    # Build model
    model = build_model(vocab_size, EMBED_SIZE, MX_SEQUENCE_LENGTH)
    
    # Fit model to data
    history = model.fit(X_train_pad, y_train,
                       epochs = args["epochs"],
                       batch_size = BATCH_SIZE,
                       validation_split = 0.1, # second validation split  
                       verbose = True)
    # Report metrics
    evaluate(model, X_test_pad, y_test_lb)
    
    # Plot history
    plot_history(history, args["epochs"])
    
    # Print info
    print("[INFO] Job complete")

# Run main() function from terminal only
if __name__ == "__main__":
    main()
