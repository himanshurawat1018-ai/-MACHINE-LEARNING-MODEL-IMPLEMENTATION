# -MACHINE-LEARNING-MODEL-IMPLEMENTATION
NAME : HIMANSHU RAWAT

INTERN ID : CT04DH2741

DOMAIN : PYTHON PROGRAMING

DURATION : 4 WEEKS

MENTOR NAME : NEELA SANTOH

DISCRIPTION ABOUT THIS PROJECT
---Project Title: Spam Email Detection using Scikit-learn---

The goal of this project is to build a machine learning model that can classify messages as either spam or ham (non-spam) using the Scikit-learn library. Spam detection is a classic natural language processing (NLP) task and has wide applications in email filtering, SMS filtering, and social media moderation. This project demonstrates how to preprocess textual data, build a classification model, evaluate its performance, and visualize results using Python-based tools and libraries.

We start by using a dataset that contains SMS messages labeled as spam or ham. For this purpose, we load the "SMS Spam Collection Dataset" available via OpenML using fetch_openml. This dataset includes thousands of messages with binary classification labels, which are essential for training a supervised learning model.

The first step involves loading the dataset into a Pandas DataFrame, renaming the columns for clarity (label and text), and analyzing the class distribution. Preprocessing includes converting categorical labels (‘spam’, ‘ham’) to numerical labels (1 and 0 respectively). The message text is stored in a feature column X and the corresponding label in y.

Text data must be converted into a numerical format before being used in machine learning models. To achieve this, we use CountVectorizer, which transforms the textual data into a sparse matrix of token counts. This "bag-of-words" representation is a standard method in text classification tasks and forms the basis for many machine learning pipelines.

Once the data is vectorized, it is split into training and test sets using train_test_split. This ensures the model is trained on one portion of the data and tested on unseen data, helping us evaluate the generalization performance of the model.

For classification, we use the Multinomial Naive Bayes algorithm, which is well-suited for text classification problems. After training the model using the training data, predictions are made on the test set. The results are then evaluated using several metrics including accuracy, classification report, and confusion matrix.

The classification report provides detailed insights into precision, recall, and F1-score for both spam and ham classes. The confusion matrix is plotted using Seaborn's heatmap, giving a visual representation of true positives, false positives, true negatives, and false negatives, which is helpful for understanding the model’s performance.

Finally, the complete implementation is written in a Jupyter Notebook with clearly labeled cells, making it easy to understand and replicate. The code can also be run in Visual Studio Code (VS Code) with the appropriate Python and Jupyter extensions.

This project not only demonstrates the use of Scikit-learn in building a predictive model but also teaches important aspects of working with text data, model evaluation, and performance visualization. It serves as a foundational project for students and developers interested in machine learning, natural language processing, and Python programming.

In summary, the project is a practical demonstration of spam detection using a real-world dataset, machine learning techniques, and Python libraries. It bridges theoretical knowledge with hands-on implementation and provides a solid introduction to classification tasks in NLP.

# OUTPUT
![Image](https://github.com/user-attachments/assets/479aef03-ea0f-41a6-9136-d279079fe515)
