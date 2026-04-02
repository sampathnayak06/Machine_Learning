import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

bc_model = DecisionTreeClassifier(max_depth=4, random_state=42)
bc_model.fit(X_cancer, y_cancer)

print("Model trained successfully.")

sample_data = pd.DataFrame({
    'cgpa': [9.2, 8.5, 9.8, 7.5, 8.2, 9.1, 7.8, 9.3, 8.4, 8.6],
    'interactivity': ['yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'yes'],
    'practical_knowledge': ['verygood', 'good', 'average', 'average', 'good', 'verygood', 'good', 'average', 'verygood', 'good'],
    'communication': ['good', 'average', 'good', 'average', 'good', 'good', 'good', 'average', 'good', 'good'],
    'job_offer': ['yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes']
})

print(sample_data.head())

label_encoders = {}
for column in ['interactivity', 'practical_knowledge', 'communication', 'job_offer']:
    le = LabelEncoder()
    sample_data[column] = le.fit_transform(sample_data[column])
    label_encoders[column] = le

X_sample = sample_data.drop('job_offer', axis=1)
y_sample = sample_data['job_offer']

sample_model = DecisionTreeClassifier(max_depth=3, random_state=42)
sample_model.fit(X_sample, y_sample)

plt.figure(figsize=(12, 8))
plot_tree(sample_model, 
          feature_names=X_sample.columns,
          class_names=label_encoders['job_offer'].classes_,
          filled=True, 
          rounded=True)
plt.title("Decision Tree for Job Offer Prediction")
plt.show()

test_sample = pd.DataFrame([{
    'cgpa': 8.5,
    'interactivity': label_encoders['interactivity'].transform(['yes'])[0],
    'practical_knowledge': label_encoders['practical_knowledge'].transform(['verygood'])[0],
    'communication': label_encoders['communication'].transform(['good'])[0]
}])

prediction = sample_model.predict(test_sample)
result = label_encoders['job_offer'].inverse_transform(prediction)
print(f"Prediction for test sample: {result[0]}")
