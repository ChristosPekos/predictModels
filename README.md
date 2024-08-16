<h1 align="center" id="title">IMDB reviews Classifiers</h1>

<p id="description">Developed two models of classifiers from scratch (Logistic Regression Naive Bayes) in order to categorize IMDB movie reviews as positive or negative. Then the capabilities of each classifier where compared to those that are available within the sklearn library.</p>

  
  
<h2>Importing the Data</h2>

*   Importing the dataset: At first the data are imported selecting the 5000 most common words and skipping the most 100 used in order to keep track of the most important words used while also skipping words like is they at etc.
*   Data transformation: Each review is transformed into a binary vector with the size being equal to the size of the vocabulary. Each index of the vectors are representing a specific word and if the value of the certain index is 1 or 0 is based on whether the word appears in the review or not

<h2>Logistic Regression</h2>

<h3>Training phase</h3>
Before the training of the Logistic Regression classifier begins the weights (symbolizing the magnitude of importance of each word) are initialized as 0. 
Then for every sample of the dataset the sigmoid calculation occurs which calculates the prediction of which class the example belongs to. 
Afterwards using Gradient Descent the weights are updated in order to find the minimum total cost of error during our predictions while also using regularization to avoid obscure weight values.
We then calculate the cost difference from before and now with the updated values. The completion of these calculations for every sample of the dataset signifies the completion of 1 epoch. 
The training will only stop on two occasions either the difference of cost from the previous iteration is less than 0.001 or we have completed 100 epochs. 
During our training we basically try to find the best possible values for the weights for each word of our vocabulary in order for our classifier to make the most accurate predictions later. 

<h3>Prediction phase</h3>
Using the sigmoid function, we calculate the probability that the review is positive with the values of weights that we obtained during the training phase. If the value that the sigmoid function produces for a review is larger than 0.5, the review
is classified as positive. Otherwise, the review is classified as negative


<h2>Naive bayes</h2> 

<h3>Training phase</h3>
During the training process of the Naive Bayes Classifier we firstly calculate the probability that a review is positive or negative just by dividing the number of positive/ negative reviews with the number of the total reviews. 
Then for every word we find all the reviews that contain the word and are positive and all the reviews that contain the word and are negative and we seperate them. After that we calculate for different probabilities: 
<ol>
  <li>The review contains the word and is positive. </li>
  <li>the review contains the word and is negative.</li> 
  <li>the review doesnt contain the word and is positive.</li>
  <li>the review doesnt contain the word and is negative.</li>
</ol>
We calculate those probabilities using the laplace rule in order to avoid any of them being 0. The probabilities that we just calculated are gonna be important during the predictions of our classifier. 

<h3>Prediction phase</h3>

<p>For each word of the prediction dataset we separate the probabilities of positive or negative reviews based on each word in two.
Afterwards, we calculate the probability that the entire review is positive or negative. We do that by:</p>

Probability of the review being positive: P(C=1|X) = P(C=1)+Σ(P(X|C=1)
*   We calculate the probability of the specific review being positive by adding the initial probability that we found about a review being positive to the sum of all probabilities of a review being positive based on every single word.

Probability of the review being negative: P(C=0|X) = P(C=0)+Σ(P(X|C=0))
*   We calculate the probability of the specific review being negative by adding the initial probability that we found about a review being negative to the sum of all probabilities of a review being negative based on every single word.
<p>We then compare these probabilities. If the first one is larger than the second one, the review is classified as positive. Otherwise, the review is classified as negative</p>

