# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import glob

def load_from_file(path):
  df = pd.read_csv(path, header=None, sep=" ")
  X = df.iloc[:, 1:257].values
  y = df.iloc[:, 0].values.astype(np.uint8)
  return X, y

import numpy as np
import struct

def read_label_file(path):
  with open(path, 'rb') as f:
    data = f.read()
    magic_number, items = struct.unpack('>II', data[0:8])
    assert(magic_number == 2049)
    y = np.frombuffer(data, dtype=np.uint8, count=items, offset=8)
    return y
    
def read_image_file(path):
  with open(path, 'rb') as f:
    data = f.read()
    magic_number, items, rows, columns = struct.unpack('>IIII', data[0:16])
    assert(magic_number == 2051)
    X = np.frombuffer(data, dtype=np.uint8, count=items*rows*columns, offset=16)
    return X.reshape((items, -1)).astype(np.float32) / 255.0 # Normalisierung auf Werte zwischen 0 und 1

"""## Allgemeine Klasse für Klassifizierer"""

class Classifier:
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def confusion_matrix(self, X, y):
        size = len(set(y))
        predicted = self.predict(X)
        
        results = np.zeros((size, size), dtype=np.int32)

        for pi, yi in zip(predicted, y):
            results[int(pi)][int(yi)] += 1

        return results

"""## Hilfsfunktionen

Als Aktivierungsfunktion benutzen wir zunächst die logistische Funktion. Später
werden wir noch ReLUs (rectified linear units) sehen.
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""Um die Bias-Einheiten einfacher zu handhaben, werden zwei Hilfsfunktionen benötigt."""

def add_ones(X):
    if len(X.shape) == 1:
        return np.hstack((X, np.ones(1)))
    else:
        return np.concatenate((X, np.ones((len(X), 1))), axis=1)

def without_ones(X):
    return X[:-1, :]

"""Da das Modell zwischen 10 Klassen unterscheiden soll, die keine natürliche Ordnung haben, müssen wir die Ausgabe als Vektor darstellen.
In der Statistik nennt man solche Variablen (labels) kategorisch.
"""

def encode_onehot(y):
    output_dimension = len(set(y))
    t = np.zeros((len(y), output_dimension))

    for i, yi in enumerate(y):
        t[i][yi] = 1.

    return t

"""Alternativ geht es mit numpy kürzer aber weniger verständlich:"""

def encode_onehot(y):
    output_dimension = len(set(y))
    return np.eye(output_dimension)[y]

"""## Gradient Descent

Es gibt viele verschiedene Optimierungsalgorithmen, die die Aktualisierungsregel anpassen. Um diese einfach austauschen zu können, abstrahieren dies die meisten ML-Bibliotheken etwas. Eine eigene Klasse sieht vielleicht zunächst übertrieben aus, es macht aber viel Sinn, wenn man etwas kompliziertere Aktualisierungsregeln betrachtet (z.B. RProp).
"""

class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    def __call__(self, weights, gradients, epoch):
        return [w - self.learning_rate * g for w,g in zip(weights, gradients)]

"""## Neuronale Netze"""

from time import time

"""Die folgende Klasse implementiert neuronale Netze mit mini-batch gradient descent. In jeder Iteration werden dabei `batch_size` viele zufällige Datenpunkte ausgesucht und durch das Netz geschickt. Bei `batch_size=1` entspricht das stochastic gradient descent (SGD), bei `batch_size=len(X)` wäre es einfach normaler batch gradient descent.

Nach jeder Epoche wird die Klassifikationsgenauigkeit ausgegeben. Man unterscheidet oft zwischen Epochen und Iterationen:
- Epochen geben an wie oft man durch den gesamten Trainingsdatensatz gegangen ist
- Iterationen sind abhängig vom konkreten Algorithmus (SGD, mini-batch -, batch gradient descent) und bezeichnen einfach die Durchläufe der jeweiligen Schleife
"""

class NeuralNetwork(Classifier):
    def __init__(self, hidden):
        self.hidden = hidden

    def fit(self, X, y, optimizer, epochs=10, test=None, batch_size=128):
        _, input_dimension = X.shape
        output_dimension = len(np.unique(y))
        self.output_dimension = output_dimension
                
        targets = encode_onehot(y)
        self._init_weights(input_dimension, output_dimension)
                
        batch_size = min(len(X), batch_size)
        
        train_history = []
        test_history = []
        
        best_test = best_iteration = 0
        num_batches = int(len(X) / batch_size)
        print("train data size: {} batch size: {} num_batches:{}".format(len(X), batch_size, num_batches))         
        
        
        for i in range(epochs):
            t = time()
            permutation = np.random.permutation(range(len(X)))
            k = 0
            for j in range(num_batches):
                if batch_size==1:
                    index = permutation[k]
                    Os = self._feed_forward(X[index])
                    gradients = self._back_propagate_single(Os, targets[index])
                    self.W = optimizer(self.W, gradients, i)                
                else:
                    subset = permutation[k:k+batch_size]
                    Os = self._feed_forward(X[subset])
                    gradients = self._back_propagate(Os, targets[subset])
                    self.W = optimizer(self.W, gradients, i)
                k += batch_size


            score = self.score(X, y)
            train_history.append(score)
            output = "Epoch %d/%d: %2.1fs – training %.5f" % (i + 1, epochs, time() - t, score)
            
            if test is not None:
                score = self.score(*test)
                test_history.append(score)
                output += " – validation %.5f" % score
                
                if score > best_test:
                    best_test = score
                    best_iteration = i
                
            print(output)
        
        if test is not None:
            print("=> Best validation accuracy was %.5f at iteration %d" % (best_test, best_iteration + 1))
                    
        return train_history, test_history
    
    def _init_weights(self, input_dimension, output_dimension):
        self.W = []
        previous_dimension = input_dimension
        
        for layer in self.hidden + [output_dimension]:
            self.W.append(self._init_layer(previous_dimension, layer))
            previous_dimension = layer
                    
    def _init_layer(self, input_dim, output_dim):
        return np.random.randn(input_dim + 1, output_dim) * np.sqrt(2.0 / (input_dim + 1 + output_dim))      
    
    def _feed_forward(self, X):
        Os = [X]
        O_last = X
        
        for Wi in self.W:
            O_hat = add_ones(O_last)
            O_last = sigmoid(O_hat.dot(Wi))
            Os.append(O_last)
            
        return Os
    
    def _back_propagate_single(self, Os, ti):
        gradients = []

        # Output layer
        o = Os[-1] # network output
        o_prev = Os[-2] # output from previous layer
        e = o - ti
        print(e)
        delta = o * (1 - o) * e
        # delta = np.diag(o * (1 - o)).dot(e)        
        gradients.append(np.outer(add_ones(o_prev), delta))
        
        # backward loop over hidden layers
        # o starts at second last, o_prev starts at third last, W starts at last weight matrix
        for o, o_prev, W in zip(Os[-2::-1], Os[-3::-1], self.W[-1::-1]):
            # delta = np.diag(o * (1 - o)).dot(without_ones(W).dot(delta))
            delta = (o * (1 - o)) * (without_ones(W).dot(delta))
            gradients.append(np.outer(add_ones(o_prev), delta))
            
        gradients.reverse()
        return gradients
    
    def _back_propagate(self, Os, ti):
        gradients = []

        # Output layer
        o = Os[-1] # network output
        print(o)
        o_prev = Os[-2] # output from previous layer
        e = o - ti      
        delta = o * (1 - o) * e
        
        # outer product for batches
        gradients.append(np.einsum("...i,...j->...ij", add_ones(o_prev), delta))
        
        # backward loop over hidden layers
        # o starts at second last, o_prev starts at third last, W starts at last weight matrix
        for o, o_prev, W in zip(Os[-2::-1], Os[-3::-1], self.W[-1::-1]):           
            # Weight matrix multiplied with delta for batches
            wd = np.einsum("...ij,...j->...i", without_ones(W), delta)
            # elementwise multiplication (broadcasting)
            delta = np.einsum("...,...", (o * (1 - o)), wd)
            # outer product for batches
            gradients.append(np.einsum("...i,...j->...ij", add_ones(o_prev), delta))
            
        # sum up all batch gradients
        gradients = [np.sum(g, axis=0) for g in reversed(gradients)]
        return gradients
    
    def predict(self, X):
        O_last = self._feed_forward(X)[-1]
        return O_last.argmax(axis=1)



"""## Visualisierung"""

def plot(history, start=0, title=""):
    acc = history[0][start:]
    val_acc = history[1][start:]
    
    x = range(1, len(acc)+1)
    
    fig = plt.figure(figsize=(25, 20))
    plt.subplot(2, 1, 1)
    
    plt.plot(x, acc, label='Training Accuracy')
    plt.plot(x, val_acc, label='Validation Accuracy')
    plt.plot([1],[1], linewidth=0.000001)
    plt.axvline(x=np.argmax(val_acc)+1, color="grey", alpha=.6, label="Best Validation", linestyle="--")
    
    plt.legend(loc='lower right')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(title)
    
    plt.xlim(1)
    plt.show()

def main():
    X_train = read_image_file("train-images.idx3-ubyte")
    y_train = read_label_file("train-labels.idx1-ubyte")
    X_test = read_image_file("train-images.idx3-ubyte")
    y_test = read_label_file("train-labels.idx1-ubyte")

    X_val = X_test[:2000]
    y_val = y_test[:2000]

    X_test = X_test[2000:]
    y_test = y_test[2000:]

    # set to True if you want to use the small digits dataset
    if False:
        X_train, y_train = load_from_file("zip.train")
        X_test, y_test = load_from_file("zip.test")
        X_val = X_test
        y_val = y_test

    print("N_train: {}  N_val: {}  N_test: {}".format(len(X_train), len(X_val), len(X_test)))

    hidden_layers = [80]
    clf = NeuralNetwork(hidden=hidden_layers)
    batch_size=128
    learning_rate = 0.01
    history = clf.fit(X_train, y_train, optimizer=GradientDescent(learning_rate=learning_rate), epochs=30, test=(X_val, y_val), batch_size=1)
    clf.score(X_test, y_test)
    plot(history, title="N={}   Batch size={}   Learning Rate={}".format(len(X_train), batch_size, learning_rate))


    hidden_layers = [70, 50]
    clf = NeuralNetwork(hidden=hidden_layers)
    batch_size=128
    learning_rate = 0.01
    history = clf.fit(X_train, y_train, optimizer=GradientDescent(learning_rate=learning_rate), epochs=30, test=(X_val, y_val), batch_size=128)
    clf.score(X_test, y_test)
    plot(history, title="N={} Batch size: {} Learning Rate: {}".format(len(X_train), batch_size, learning_rate))

if __name__ == "__main__":
    main()
