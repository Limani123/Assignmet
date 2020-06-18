# Assignmet
NLP assignment 

The following code is the assignment solution.

The given assignment is based on text classification so we are using Universal Sentence Encode as it encodes text into high dimensional vectors that can be used for this task and it also capture meaning of sentences using sentence embeddings. We are using pre-trained model here and fine tuning it on our own dataset. As we are not trainig it from scratch hence it will take less time in training.


Requirements:
tensorflow,
pandas,
sklearn,
numpy

dataset:
we are using the dataset that was given in the assignment. We have already converted the text file to csv file and using the csv file in this solution.

dataset distribution:
We are spliting the dataset into training, testing and validation set. 

Accuracy
We are getting above 90% accuracy on different distribution of dataset.


To run a code:
put dataset file in the same folder as the code then in the terminal type the below command:
python text_classification.py.
At the end it is also asking for user input to classify the user sentence. Here it using the saved model used in the code to predict the result.


