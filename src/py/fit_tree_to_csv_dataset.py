import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load CSV dataset
df = pd.read_csv(r"C:\Users\79140\PycharmProjects\procedural-terrain-generation\data\terrain_features_dataset.csv")

# Separate features and label
X = df.drop("realness", axis=1)
y = df["realness"]

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=421)

# Fit regression tree
model = DecisionTreeRegressor(max_depth=5, random_state=421)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R^2 accuracy: {r2:.3f}")

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Regressor (max_depth=5)")
plt.show()
