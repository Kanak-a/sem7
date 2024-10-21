import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

# Convert labels to binary values: spam = 1, ham = 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Splitting the dataset into messages and labels
X = df['message']
y = df['label']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorizing the text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Making predictions
y_pred = classifier.predict(X_test_tfidf)

# Evaluating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Test prediction on a new email
new_email = ["Congratulations! You have won a free vacation! Claim now"]
new_email_tfidf = vectorizer.transform(new_email)
prediction = classifier.predict(new_email_tfidf)
print("Spam" if prediction[0] else "Not Spam")
