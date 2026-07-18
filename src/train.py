from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data 
y = iris.target
print(iris.feature_names, iris.target_names) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Prediction:", y_pred[:5])
print("True labels:", y_test[:5])

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

import joblib
joblib.dump(model, "outputs/model.joblib")
print("Model saved successfully!")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("Iris Classification Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()
print("Confusion matrix saved successfully!")