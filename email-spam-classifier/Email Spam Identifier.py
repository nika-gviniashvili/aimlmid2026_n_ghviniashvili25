import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

#Load the the data , our csv file

def load_data(csv_path):
    data = pd.read_csv(csv_path)
    X = data.drop(columns=['is_spam'])
    y = data['is_spam']
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

#Model evaluation

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Save confusion matrix heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Legit','Spam'], yticklabels=['Legit','Spam'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")

# Feature extraction from email text

SPAM_WORDS = [
    "free", "win", "winner", "cash", "prize",
    "money", "offer", "urgent", "click"
]

def extract_features(email_text):
    words = email_text.split()
    word_count = len(words)
    link_count = len(re.findall(r"http[s]?://", email_text))
    capital_words = sum(1 for w in words if w.isupper())
    spam_word_count = sum(1 for w in words if w.lower() in SPAM_WORDS)
    return np.array([[word_count, link_count, capital_words, spam_word_count]])


#Predict a new email

def classify_email(model):
    print("\nEnter email text (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    email_text = " ".join(lines)
    features = extract_features(email_text)
    prediction = model.predict(features)

    if prediction[0] == 1:
        print("\nResult: SPAM EMAIL")
    else:
        print("\nResult: LEGITIMATE EMAIL")

# Visualizations
def visualize_data(X, y, model):
    data = X.copy()
    data['is_spam'] = y

    spam_counts = data['is_spam'].value_counts()
    plt.figure(figsize=(6,4))
    plt.bar(['Legitimate','Spam'], spam_counts, color=['green','red'])
    plt.title("Distribution of Email Types")
    plt.ylabel("Number of Emails")
    plt.xlabel("Email Type")
    plt.savefig("class_distribution.png")
    plt.close()
    print("Class distribution saved as 'class_distribution.png'")
    plt.figure(figsize=(6,4))
    plt.bar(X.columns, model.coef_[0], color='skyblue')
    plt.title("Feature Importance (Logistic Regression Coefficients)")
    plt.ylabel("Coefficient Value")
    plt.xlabel("Feature")
    plt.savefig("feature_importance.png")
    plt.close()
    print("Feature importance saved as 'feature_importance.png'")


# Main application
def main():
    # Ask for CSV path
    csv_path = input("Enter path to CSV file: ").strip()
    if not os.path.isfile(csv_path):
        print("Error: File not found!")
        return

    X, y = load_data(csv_path)
    model, X_test, y_test = train_model(X, y)

    print("\nModel Coefficients:")
    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"{feature}: {coef:.4f}")


    evaluate_model(model, X_test, y_test)


    visualize_data(X, y, model)

    classify_email(model)

if __name__ == "__main__":
    main()
