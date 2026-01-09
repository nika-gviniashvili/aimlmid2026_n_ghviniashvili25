AI and ML for Cybersecurity
Midterm Exam

Spam email detection
Task1. I have uploaded the provided CSV file n_ghviniashvili25_15289.csv to my GitHub repository which I have used as an input for my python application.The link to the uploaded file is: https://github.com/nika-gviniashvili/aimlmid2026_n_ghviniashvili25/tree/main/email-spam-classifier

Task2. Logistic Regression Model
a) Data Loading and Processing
I wrote a function to load the CSV file using pandas. I separated the features (words, links, capital_words, spam_word_count) into X and the target is_spam into y. Then I split the dataset into 70% training and 30% testing using scikit-learn’s train_test_split:

def load_data(csv_path):
    data = pd.read_csv(csv_path)
    X = data.drop(columns=['is_spam'])
    y = data['is_spam']
    return X, y
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
 
b) Training the Logistic Regression Model
I trained the Logistic Regression model with max_iter=1000 to make sure it converged, I chose Logistic Regression because this is a binary classification problem: spam vs. legitimate emails:

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

c) Model Coefficients
After training, I printed the coefficients learned by the model:

Model Coefficients:
words: 0.0084
links: 0.8977
capital_words: 0.4687
spam_word_count: 0.8528


Task 3: Model Validation
I evaluated the model using the test dataset (30% of the data not used for training). I calculated the confusion matrix and accuracy:

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)


| Legitimate       | 348        | 14   |
| Spam             | 17         | 371  |

accuracy was 0.9587:

I also saved a heatmap of the confusion matrix using seaborn for visualization: confusion_matrix.png.  The confusion matrix shows correct and incorrect classifications.Accuracy indicates that almost 95.87% of emails were classified correctly.

Task 4: Email Classification
I added a feature to my console app to allow manual email input. The program extracts features from the text:
* Number of words
* Number of links
* Number of fully capitalized words
* Number of spam words from the list
  
  SPAM_WORDS =
  [
    "free", "win", "winner", "cash", "prize",
    "money", "offer", "urgent", "click"
]  

The trained model predicts whether the email is spam or legitimate based on these features.


Task 5: Spam Email Example
I manually composed an email designed to be classified as spam and as shown result was SPAM EMAIL because email text contains multiple spam words: win, cash, prize, money, urgent, offer, click, free, winner and it has link too:


<img width="959" height="316" alt="text results-spam" src="https://github.com/user-attachments/assets/d714a6e6-02d7-4e0c-b319-db6b951057e9" />


Task 6: Legitimate Email Example
I also composed an email to be classified as legitimate and as it is shown it was classified as LEGITIMATE EMAIL because it did not contain spam words and also did not had link, there was not fully capitalized words and it was written in a normal language:


<img width="1002" height="768" alt="text results-legitimate" src="https://github.com/user-attachments/assets/e09dc221-f608-46ef-84e0-bb980b05ee8e" />



Task 7: Visualizations
for visualizations I have created a bar chart to show the number of spam and legitimate emails and a bar chart to show the influence of each feature:


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

After running the code these png files will be generated.


