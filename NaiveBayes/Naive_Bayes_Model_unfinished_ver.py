from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# load our refined dataset for Naive Bayes:
ref_df = pd.read_csv("/Users/evermore/Downloads/BBBP_top_predictors.csv")

# identify the features:
feature_names = ref_df.columns[:-1]
label_name = ref_df.columns[-1]

print("Features:", feature_names)
print("Label:", label_name)

# split into features and label:
X = ref_df[feature_names]
y = ref_df[label_name]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train Naive Bayes using three different priors:
for p in [0.5,0.6,0.7]:
    nb = GaussianNB(priors=[1-p,p])
    nb.fit(X_train,y_train)
    y_pred = nb.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    print(p, acc)

priors = [0.5, 0.6, 0.7]

train_acc = []
test_acc = []

for p in priors:
    nb = GaussianNB(priors=[1-p, p])
    nb.fit(X_train, y_train)

    # train accuracy
    y_train_pred = nb.predict(X_train)
    train_acc.append(accuracy_score(y_train, y_train_pred))

    # test accuracy
    y_test_pred = nb.predict(X_test)
    test_acc.append(accuracy_score(y_test, y_test_pred))

# plot
plt.figure(figsize=(6,4))
plt.plot(priors, train_acc, marker='o', label="Train Accuracy")
plt.plot(priors, test_acc, marker='o', label="Test Accuracy")
plt.xlabel("Prior P(y=1)")
plt.ylabel("Accuracy")
plt.title("Effect of Class Prior on GaussianNB Performance")
plt.legend()
plt.grid(True)
plt.show()

# Experiment: test accuracy with balanced class samples:








