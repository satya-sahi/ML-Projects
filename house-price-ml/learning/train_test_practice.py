"""
Practice with Train/Test Split and Overfitting
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import numpy as np

# Create simple dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])  # y = 2*X

print("=== Original Data ===")
print(f"X: {X.flatten()}")
print(f"y: {y}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train overfitted model (max_depth=None = memorizes)
overfit_model = DecisionTreeRegressor(max_depth=None, random_state=42)
overfit_model.fit(X_train, y_train)

train_r2_overfit = r2_score(y_train, overfit_model.predict(X_train))
test_r2_overfit = r2_score(y_test, overfit_model.predict(X_test))

print("\n=== Overfitted Model (max_depth=None) ===")
print(f"Training R²: {train_r2_overfit:.3f}")
print(f"Test R²: {test_r2_overfit:.3f}")
print(f"Gap: {train_r2_overfit - test_r2_overfit:.3f}")

# Train good model (max_depth=2 = generalizes)
good_model = DecisionTreeRegressor(max_depth=2, random_state=42)
good_model.fit(X_train, y_train)

train_r2_good = r2_score(y_train, good_model.predict(X_train))
test_r2_good = r2_score(y_test, good_model.predict(X_test))

print("\n=== Good Model (max_depth=2) ===")
print(f"Training R²: {train_r2_good:.3f}")
print(f"Test R²: {test_r2_good:.3f}")
print(f"Gap: {train_r2_good - test_r2_good:.3f}")

print("\n💡 Smaller gap = less overfitting!")
print("💡 Good model: train and test R² are similar")
print("💡 Overfit model: train R² much higher than test R²")
