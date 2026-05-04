# Import required libraries from sklearn.datasets 
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report

# Load dataset 
digits = load_digits() 

# Feature matrix (pixel values) 
X = digits.data 

# Target labels (0–9) 
y = digits.target

# Split dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 ) 

# Create classifier 
model = SVC(gamma=0.001) 

# Train the model 
model.fit(X_train, y_train) 

# Predict
y_pred = model.predict(X_test) 

# Accuracy 
print("Accuracy:", accuracy_score(y_test, y_pred)) 

# Detailed report 
print(classification_report(y_test, y_pred))