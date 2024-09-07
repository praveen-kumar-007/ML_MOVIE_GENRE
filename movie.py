import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


train_data = pd.read_csv('train_data.txt', delimiter='\t', header=None, names=['Plot', 'Genre'])
test_data = pd.read_csv('test_data.txt', delimiter='\t', header=None, names=['Plot'])
test_data_solution = pd.read_csv('test_data_solution.txt', delimiter='\t', header=None, names=['Genre'])

print("First few rows of the training data:")
print(train_data.head())

print("\nFirst few rows of the test data:")
print(test_data.head())

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(train_data['Plot'])
X_test_tfidf = tfidf.transform(test_data['Plot'])

y_train = train_data['Genre']
y_test = test_data_solution['Genre']

X_train, X_val, y_train, y_val = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_val)

print("\nNaive Bayes Results:")
print('Accuracy:', accuracy_score(y_val, y_pred_nb))
print('Confusion Matrix:\n', confusion_matrix(y_val, y_pred_nb))
print('Classification Report:\n', classification_report(y_val, y_pred_nb))

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)

print("\nLogistic Regression Results:")
print('Accuracy:', accuracy_score(y_val, y_pred_lr))
print('Confusion Matrix:\n', confusion_matrix(y_val, y_pred_lr))
print('Classification Report:\n', classification_report(y_val, y_pred_lr))

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_val)

print("\nSupport Vector Machine Results:")
print('Accuracy:', accuracy_score(y_val, y_pred_svm))
print('Confusion Matrix:\n', confusion_matrix(y_val, y_pred_svm))
print('Classification Report:\n', classification_report(y_val, y_pred_svm))


y_test_pred = lr.predict(X_test_tfidf)

print("\nTest Data Results with Logistic Regression:")
print('Accuracy:', accuracy_score(y_test, y_test_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_test_pred))
print('Classification Report:\n', classification_report(y_test, y_test_pred))
