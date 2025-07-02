#                                                                       importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

#                                                               loading the dataset into a dataframe
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

#                                                                          performing eda

#basic checks
print(iris_df.head())
print(iris_df.tail())
print(iris_df.info())
print(iris_df.describe())
#seeing if any nulls
print("Null values:\n",iris_df.isnull().sum())

#                                                                         data visualization

#1. scatterplots for sepal and petal sizes
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Define the color map for each species
species_names = iris.target_names
colors = ['red', 'green', 'blue']

# First subplot:Sepal
for i, species in enumerate(species_names):
    subset = iris_df[iris_df['species'] == i]
    axs[0].scatter(subset["sepal length (cm)"], subset["sepal width (cm)"],
                   color=colors[i], label=species)

axs[0].set_title("Iris species by sepal length and width")
axs[0].set_xlabel("Sepal length (cm)")
axs[0].set_ylabel("Sepal width (cm)")
axs[0].legend()

# Second subplot:Petal
for i, species in enumerate(species_names):
    subset = iris_df[iris_df['species'] == i]
    axs[1].scatter(subset["petal length (cm)"], subset["petal width (cm)"],
                   color=colors[i], label=species)

axs[1].set_title("Iris species by petal length and width")
axs[1].set_xlabel("Petal length (cm)")
axs[1].set_ylabel("Petal width (cm)")
axs[1].legend()

#2. boxplots for sepal and petal sizes
fig, axs1 = plt.subplots(2, 2)

#fig 1
x_setosa_sepal = iris_df[iris_df['species'] == 0]['sepal length (cm)']
x_versicolor_sepal = iris_df[iris_df['species'] == 1]['sepal length (cm)']
x_virginica_sepal = iris_df[iris_df['species'] == 2]['sepal length (cm)']
axs1[0,0].boxplot([x_setosa_sepal,x_versicolor_sepal,x_virginica_sepal])
axs1[0,0].set_title("sepal length by species")
axs1[0, 0].set_xticklabels(['Setosa', 'Versicolor', 'Virginica'])
#fig2
y_setosa_sepal = iris_df[iris_df['species'] == 0]['sepal width (cm)']
y_versicolor_sepal = iris_df[iris_df['species'] == 1]['sepal width (cm)']
y_virginica_sepal = iris_df[iris_df['species'] == 2]['sepal width (cm)']
axs1[0,1].boxplot([y_setosa_sepal,y_versicolor_sepal,y_virginica_sepal])
axs1[0,1].set_title("sepal width by species")
axs1[0, 1].set_xticklabels(['Setosa', 'Versicolor', 'Virginica'])
#fig3
x_setosa_petal = iris_df[iris_df['species'] == 0]['petal length (cm)']
x_versicolor_petal = iris_df[iris_df['species'] == 1]['petal length (cm)']
x_virginica_petal = iris_df[iris_df['species'] == 2]['petal length (cm)']
axs1[1,0].boxplot([x_setosa_petal,x_versicolor_petal,x_virginica_petal])
axs1[1,0].set_title("Petal length by species")
axs1[1, 0].set_xticklabels(['Setosa', 'Versicolor', 'Virginica'])
#fig4
y_setosa_petal = iris_df[iris_df['species'] == 0]['petal width (cm)']
y_versicolor_petal = iris_df[iris_df['species'] == 1]['petal width (cm)']
y_virginica_petal = iris_df[iris_df['species'] == 2]['petal width (cm)']
axs1[1,1].boxplot([y_setosa_petal,y_versicolor_petal,y_virginica_petal])
axs1[1,1].set_title("Petal width by species")
axs1[1,1].set_xticklabels(['Setosa', 'Versicolor', 'Virginica'])


plt.tight_layout()
plt.show()


#                                                             model training and prediction
X,y=load_iris(return_X_y=True)
# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier()
}

best_score = 0
best_model_name = ''
best_pipeline = None

#preprocessing
# Loop through models
for name, model in models.items():
    steps = []
    if name != 'Decision Tree':  # Only scale if needed
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', model))
    
    pipeline = Pipeline(steps)
    scores = cross_val_score(pipeline, X, y, cv=5)
    
    print(f"\n{name} Cross-validation scores: {scores}")
    print(f"{name} Average CV score: {scores.mean():.3f}")
    
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_model_name = name
        best_pipeline = pipeline

# Fit best model on full data
best_pipeline.fit(X, y)

# Predict new data
new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = best_pipeline.predict(new_data)
print(f"\nBest model: {best_model_name} with accuracy {best_score:.3f}")
predicted_species = iris.target_names[prediction[0]]
print("Predicted species:", predicted_species,"\n")

#metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X)
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred, target_names=iris.target_names))
