import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import math
from tabulate import tabulate


# create vocabulary, we skip top 100 most frequent words and get only the rest 5000 frequent words and also skipping the rest frequent
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words = 5000, skip_top = 100 )
word_index = tf.keras.datasets.imdb.get_word_index()
index2word = dict((i + 3, word) for (word, i) in word_index.items())
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'
x_train = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train])
x_test = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test])

#create binary vectors, 1 if the word appears in the review 0 if i doesn't 
binary_vectorizer = CountVectorizer(binary=True)
x_train_binary = binary_vectorizer.fit_transform(x_train)
x_test_binary = binary_vectorizer.transform(x_test)
x_train_binary = x_train_binary.toarray()
x_test_binary = x_test_binary.toarray()

class NaiveBayes:
  def fit(self, x, y):
        
    # calculate the log probability for C =1 and C = 0 ( c is 1 if the review is good and 0 if it is bad)
      self.prob_c1 = math.log(np.count_nonzero(y)/len(y))
      self.prob_c0 = math.log(1-(np.count_nonzero(y)/len(y)))

    # select only the x=1 for the c=1. It means the review is good (1) and it contains the x word (1) 
      x1c1 = [x[i] for i in range(len(y)) if y[i] == 1]
      x1c1 = np.array(x1c1)

    # select only the x=1 for the c=0 . It means the review is bad (0) and it contains the x word (1) 
      x1c0 = [x[i] for i in range(len(y)) if y[i] == 0]
      x1c0 = np.array(x1c0)

    # calculate the log conditional probabilities P(X=1| C=1), P(X=1| C=0), P(X=0| C=1) = 1 - P(X=1| C=1), P(X=0| C=0) = 1 - P(X=1| C=0). In all probabilitys we have added the laplace rule
      self.prob_x1c1 = np.log((np.count_nonzero(x1c1, axis=0) + 1) / (np.count_nonzero(y) + 2))
      self.prob_x1c0 = np.log((np.count_nonzero(x1c0, axis=0) + 1) / (len(y) - np.count_nonzero(y) + 2))
      self.prob_x0c1 = np.log(1-((np.count_nonzero(x1c1, axis=0) + 1) / (np.count_nonzero(y) + 2)))
      self.prob_x0c0 = np.log(1-((np.count_nonzero(x1c0, axis=0) + 1) / (len(y) - np.count_nonzero(y) + 2)))


  def predict(self, x):
      
    #if Xi=1 select P(Xi=1|c=1) (or P(Xi=1|c=0)) else if Xi=0 select P(Xi=0|c=1) (or P(Xi=0|c=0) )
    temp1 = np.where(x==1, self.prob_x1c1, self.prob_x0c1)
    temp2 = np.where(x==1, self.prob_x1c0, self.prob_x0c0)
       
    # calculate the P(C=1|X). We used logs so our type is P(C=1|X) = P(C=1)+Σ(P(X|C=1)) which are calculated using logs as described above.
    prob_c1X = self.prob_c1 + np.sum(temp1, axis = 1)  #calculate the sum for each row of temp 1
    #same as above but for P(C=0|X)
    prob_c0X = self.prob_c0 + np.sum(temp2, axis = 1)
       
    #create a predictions numpy array that contains 1 if review is good based on (P(C=1|X)>P(C=0|X)) or else 0
    predictions = np.where(prob_c1X > prob_c0X, 1, 0)
    return predictions

    




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
  bayes = classifier

  #split the train data into groups
  for i in range(0,len(x_splits)):
    if (i==0):
      curr_x = x_splits[0]
      curr_y = y_splits[0]
    else:
      curr_x = np.concatenate((curr_x, x_splits[i]), axis=0)  # reunite the current group with the previous
      curr_y = np.concatenate((curr_y, y_splits[i]), axis=0)
    bayes.fit(curr_x,curr_y)  # call train method 
    train_predict = bayes.predict(curr_x)   #call predict method on training data
    test_predict = bayes.predict(x_test)    #call predict method on test data
      

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
    
naive_bayes = NaiveBayes()
scores1 = list(scores(naive_bayes,x_train_binary,y_train,x_test_binary,y_test,5))

#create the curves
titles = ["Accuracy","Precision","Recall","F1"]
j=0
for i in range(0,7,2):
    createCurve(scores1[i], scores1[i+1], len(x_train_binary), scores1[8], titles[j])
    j = j+1

#creates the table 
    
createTable(scores1)

#compares with scikit learn Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
scores2 = list(scores(nb,x_train_binary,y_train,x_test_binary,y_test,5))
j=0
for n in range(0,7,2):
    compareCurves(scores1[n], scores1[n+1], scores2[n], scores2[n+1], len(x_train_binary), scores1[8], titles[j])
    j = j+1

compareTables(scores1,scores2,scores1[8])

