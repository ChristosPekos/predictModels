import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import math
from tabulate import tabulate
from sklearn.utils import shuffle



# create vocabulary, we skip top 100 most frequent words and get only the rest 5000 frequent words and also skipping the rest frequent

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words = 5000, skip_top = 100)
x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1)
word_index = tf.keras.datasets.imdb.get_word_index()
index2word = dict((i + 3, word) for (word, i) in word_index.items())
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'
x_train = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train])
x_dev = np.array([' '.join([index2word[idx] for idx in text]) for text in x_dev])
x_test = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test])


#create binary vectors, 1 if the word appears in the review 0 if i doesn't 
binary_vectorizer = CountVectorizer(binary=True)
x_train_binary = binary_vectorizer.fit_transform(x_train)
x_dev_binary = binary_vectorizer.transform(x_dev)
x_test_binary = binary_vectorizer.transform(x_test)
x_train_binary = x_train_binary.toarray()
x_dev_binary = x_dev_binary.toarray()
x_test_binary = x_test_binary.toarray()


# add 1s for each row as the first column representing X0 = 1
x_train_binary = np.insert(x_train_binary, 0, 1, axis=1)
x_dev_binary = np.insert(x_dev_binary, 0, 1, axis=1)
x_test_binary = np.insert(x_test_binary, 0, 1, axis=1)


def sigmoid(t):
  return 1.0 / (1.0 + np.exp(-t))


class LogisticRegression:

    def __init__(self, reg_value, learn_rate):
      self.reg_value = reg_value
      self.learn_rate = learn_rate

    def fit(self, x, y):
      epochs = 0
      temp_list = list()
      current_loss = 0
      self.weight = np.zeros(x.shape[1])

      x,y = shuffle(x,y, random_state=0)

      while(epochs < 100):

        for i in range(x.shape[0]):
          temp_list.append(np.dot(self.weight,x[i]))
          sigmoid_calc = sigmoid(temp_list[i])

          self.weight = (1 - 2 * self.reg_value * self.learn_rate) * self.weight + self.learn_rate * x[i] * (y[i]-sigmoid_calc)

        previous_loss = current_loss
        current_loss = (1/(x.shape[0]))*(np.sum(y * np.log((sigmoid(np.dot(x, self.weight))) + 1e-100) + (1 - y) * np.log(1 - sigmoid(np.dot(x, self.weight)) + 1e-100)))

        if epochs>1 and abs(current_loss - previous_loss) < 0.001:
            print("Training ended after " ,epochs + 1, " epochs")
            break
        elif epochs == 99:
          print( "Training ended after 100 epochs")
        epochs+=1


    def predict(self, x):
      temp_list = list()

      for k in range (x.shape[0]):
        prediction_score = np.dot(self.weight, x[k])
        if sigmoid(prediction_score) <= 0.5:
          temp_list.append(0)
        else:
          temp_list.append(1)

      return temp_list
  

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#calculates the necessary scores for eavluation

def scores(classifier, x_train,y_train,x_test,y_test,n_splits):
  split_size = int(len(x_train) / n_splits)
  x_splits = np.split(x_train, n_splits) # must be equal division
  y_splits = np.split(y_train, n_splits)
  train_accuracies = list()
  train_f1 = list()
  train_precision = list()
  train_recall = list()
  test_accuracies = list()
  test_f1 = list()
  test_precision = list()
  test_recall = list()
  log_regression = classifier

    #split the train data into groups

  for i in range(0,len(x_splits)):
    if (i==0):
      curr_x = x_splits[0]
      curr_y = y_splits[0]
    else:
      curr_x = np.concatenate((curr_x, x_splits[i]), axis=0)  # reunite the current group with the previous
      curr_y = np.concatenate((curr_y, y_splits[i]), axis=0)

    log_regression.fit(curr_x,curr_y) # call train method 
    train_predict = log_regression.predict(curr_x)  #call predict method on training data
    test_predict = log_regression.predict(x_test)   #call predict method on test data



  #append each value to its list
    train_accuracies.append(accuracy_score(curr_y,train_predict))
    test_accuracies.append(accuracy_score(y_test, test_predict))

    train_f1.append(f1_score(curr_y,train_predict))
    test_f1.append(f1_score(y_test, test_predict))

    train_precision.append(precision_score(curr_y,train_predict))
    test_precision.append(precision_score(y_test, test_predict))

    train_recall.append(recall_score(curr_y,train_predict))
    test_recall.append(recall_score(y_test, test_predict))

  return train_accuracies,test_accuracies,train_precision,test_precision,train_recall,test_recall,train_f1,test_f1, split_size


# creates the curve based on its title 
def createCurve(train_score,test_score, x_len, split_size, title):
  plt.plot(list(range(split_size, x_len + split_size,
                    split_size)), train_score, 'o-', color="b",
            label="Training "+ title)
  plt.plot(list(range(split_size, x_len + split_size,
                    split_size)), test_score, 'o-', color="red",
          label="Testing "+ title)
  plt.legend(loc="lower right")
  plt.xlabel('Percentage of data')
  plt.ylabel(title)
  plt.show()



#creates a table containing all evaluation scores
def createTable(score_list1):
  data = {
    'Train Accuracy': score_list1[0],
    'Test Accuracy': score_list1[1],
    'Train Precision': score_list1[2],
    'Test Precision': score_list1[3],
    'Train Recall': score_list1[4],
    'Test Recall': score_list1[5],
    'Train F1-score': score_list1[6],
    'Test F1-score': score_list1[7]
  }

  split_size = score_list1[8]

  indexes = list(range(split_size, len(x_train) + split_size, split_size))
  print(tabulate(data,headers='keys', tablefmt='fancy_grid', showindex=indexes))



#creates the curves of two different classifiers 
def compareCurves(train_score1,test_score1,train_score2,test_score2, x_len, split_size, title):
  plt.plot(list(range(split_size, x_len + split_size,
                      split_size)), train_score1, 'o-', color="b",
             label="My Training "+ title)
  plt.plot(list(range(split_size, x_len + split_size,
                      split_size)), test_score1, 'o-', color="red",
           label="My Testing "+ title)
  plt.plot(list(range(split_size, x_len + split_size,
                      split_size)), train_score2, 'o-', color="green",
             label="Sklearn Training "+ title)
  plt.plot(list(range(split_size, x_len + split_size,
                      split_size)), test_score2, 'o-', color="purple",
           label="Sklearn Testing "+ title)
  plt.legend(loc="lower right")
  plt.xlabel('Percentage of data')
  plt.ylabel(title)
  plt.show()



#creates a table containing scores from both classifiers
def compareTables(score_list1,score_list2, split_size):
  data = {
    'My Train Accuracy': score_list1[0],
    'Sklearn Train Accuracy': score_list2[0],
    'My Test Accuracy': score_list1[1] ,
    'Sklearn Test Accuracy': score_list2[1] ,
    'My Train Precision': score_list1[2],
    'Sklearn Train Precision': score_list2[2],
    'My Test Precision': score_list1[3],
    'Sklearn Test Precision': score_list2[3],
    'My Train Recall': score_list1[4],
    'Sklearn Train Recall': score_list2[4],
    'My Test Recall': score_list1[5],
    'Sklearn Test Recall': score_list2[5],
    'My Train F1-score': score_list1[6],
    'Sklearn Train F1-score': score_list2[6],
    'My Test F1-score': score_list1[7],
    'Sklearn Test F1-score': score_list2[7]

    }

  indexes = list(range(split_size, len(x_train) + split_size, split_size))

  print(tabulate(data,headers='keys', tablefmt='fancy_grid', showindex=indexes))




                                # MAIN
  


reg_param = [0.01,0.001,0.0001]  # l
learning_rate = [0.01,0.001,0.0001]  # h
best_reg = None
best_lr = None
best_accuracy = 0.0
current_accuracy = 0.0
predictions = None

#find best hyperparameters (learning rate and regularization value)
for reg_p in reg_param:
  for lr in learning_rate:
    log_reg = LogisticRegression(reg_p, lr)
    log_reg.fit(x_train_binary, y_train) #train on training data
    predictions = log_reg.predict(x_dev_binary) #test on development data 
    current_accuracy = accuracy_score(y_dev, predictions)
    print("With regularization parameter: ", reg_p, " and learning rate: ", lr, " the accuracy is: ", current_accuracy)
    if (current_accuracy > best_accuracy):      #find best accuracy
      best_accuracy = current_accuracy
      best_reg = reg_p
      best_lr = lr
print("Best accuracy is: ",best_accuracy," with best regularization parameter: ", best_reg," and best learning rate: ", best_lr )




# call logistic regression using best hyperparameters

log_reg = LogisticRegression(best_reg, best_lr)
scores1 = list(scores(log_reg,x_train_binary,y_train,x_test_binary,y_test,5))
titles = ["Accuracy","Precision","Recall","F1"]
j=0

#create the curves

for i in range(0,7,2):
    createCurve(scores1[i], scores1[i+1], len(x_train_binary), scores1[8], titles[j])
    j = j+1

#creates the table 
    
createTable(scores1)


#compares with scikit learn logistic regression with stochastic gradient descent 
from sklearn.linear_model import SGDClassifier
log_sgd = SGDClassifier(loss='log')
scores2 = list(scores(log_sgd,x_train_binary,y_train,x_test_binary,y_test,5))
j=0
for n in range(0,7,2):
    compareCurves(scores1[n], scores1[n+1], scores2[n], scores2[n+1], len(x_train_binary), scores1[8], titles[j])
    j = j+1

compareTables(scores1,scores2,scores1[8])


