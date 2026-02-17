"""
Training script for house price prediction model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
from pathlib import Path
import json

from preprocessing import HousePricePreprocessor


def train_model(data_path="data/houses.xlsx", test_size=0.2, random_state=42):
    """
    Train house price prediction model

    Args:
        data_path: Path to Excel file with training data
        test_size: Fraction of data to use for testing (0.2 = 20%)
        random_state: Random seed for reproducibility
    """
    print("🚀 Starting model training...")
    print(f"Data path: {data_path}")
    print(f"Test size: {test_size * 100}%\n")

    # Load data
    df = pd.read_excel(data_path)
    print(f"✅ Loaded {len(df)} properties\n")

    # Preprocess data
    print("🔧 Preprocessing data...")
    preprocessor = HousePricePreprocessor()
    X, y = preprocessor.fit_transform(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("\n📊 Data split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")

    # Train model
    print("🤖 Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,  # Number of trees
        max_depth=10,  # Maximum depth of trees
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
    )

    model.fit(X_train, y_train)
    print("✅ Model trained!\n")

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate on training set
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Evaluate on test set
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    # Print results
    print("=" * 60)
    print("📈 MODEL EVALUATION RESULTS")
    print("=" * 60)
    print("\n🎯 Training Set Performance:")
    print(f"  MAE (Mean Absolute Error):  ${train_mae:,.0f}")
    print(f"  RMSE (Root Mean Squared):   ${train_rmse:,.0f}")
    print(f"  R² Score:                   {train_r2:.3f}")

    print("\n🎯 Test Set Performance:")
    print(f"  MAE (Mean Absolute Error):  ${test_mae:,.0f}")
    print(f"  RMSE (Root Mean Squared):   ${test_rmse:,.0f}")
    print(f"  R² Score:                   {test_r2:.3f}")

    print("\n💡 Interpretation:")
    print(f"  - On average, predictions are off by ${test_mae:,.0f}")
    print(f"  - Model explains {test_r2 * 100:.1f}% of price variance")

    # Warning about small dataset
    if len(df) < 100:
        print("\n⚠️  WARNING: Small dataset detected!")
        print(f"  Current samples: {len(df)}")
        print("  Recommended: 500-1000+ samples for reliable model")
        print("  This model is for LEARNING purposes only")

    # Check for overfitting
    if train_r2 - test_r2 > 0.3:
        print("\n⚠️  OVERFITTING DETECTED!")
        print(f"  Training R²: {train_r2:.3f}")
        print(f"  Test R²: {test_r2:.3f}")
        print(f"  Difference: {train_r2 - test_r2:.3f}")
        print("  Solution: Get more data or simplify model")

    print("\n" + "=" * 60)

    # Feature importance
    feature_importance = pd.DataFrame(
        {
            "feature": preprocessor.feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\n🔝 Top 5 Most Important Features:")
    print(feature_importance.head().to_string(index=False))

    # Save model and preprocessor
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / f"model_{timestamp}.pkl"
    preprocessor_path = model_dir / f"preprocessor_{timestamp}.pkl"

    joblib.dump(model, model_path)
    preprocessor.save(preprocessor_path)

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "train_mae": float(train_mae),
        "train_rmse": float(train_rmse),
        "train_r2": float(train_r2),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
        "features": preprocessor.feature_columns,
        "model_params": model.get_params(),
    }

    metadata_path = model_dir / f"metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n💾 Model saved:")
    print(f"  Model: {model_path}")
    print(f"  Preprocessor: {preprocessor_path}")
    print(f"  Metadata: {metadata_path}")

    # Show sample predictions
    print("\n🎲 Sample Predictions vs Actual:")
    print("-" * 60)
    comparison = pd.DataFrame(
        {
            "Actual": y_test.values[:5],
            "Predicted": y_test_pred[:5],
            "Difference": y_test.values[:5] - y_test_pred[:5],
        }
    )
    print(comparison.to_string(index=False))

    return model, preprocessor, metadata


if __name__ == "__main__":
    # Train model
    model, preprocessor, metadata = train_model()

    print("\n✅ Training complete! Ready for Day 3 (API development)")
