from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import pandas as pd
import numpy as np


class ModelTrainer:

    def __init__(self, X, y, extractor):

        self.X = X
        self.y = y
        self.extractor = extractor

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.models = {}

    def train_linear_regression(self):

        model = LinearRegression()
        model.fit(self.X_train, self.y_train)

        self.models["linear_regression"] = model
        self._evaluate_model(model, "Linear Regression")

        return model

    def train_random_forest(self):

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(self.X_train, self.y_train)

        self.models["random_forest"] = model
        self._evaluate_model(model, "Random Forest")

        return model

    def train_gradient_boosting(self):

        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )

        model.fit(self.X_train, self.y_train)

        self.models["gradient_boosting"] = model
        self._evaluate_model(model, "Gradient Boosting")

        return model

    def _evaluate_model(self, model, model_name):

        y_pred = model.predict(self.X_test)

        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"\n{model_name} Results:")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"MAE:  ${mae:,.2f}")
        print(f"R²:   {r2:.4f}")

        cv_scores = cross_val_score(
            model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="r2"
        )

        print(f"Cross-Val R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    def save_pipeline(self, model_name, filename):

        pipeline = {
            "model": self.models[model_name],
            "vectorizer": self.extractor.vectorizer,
            "scaler": self.extractor.scaler,
            "label_encoder": self.extractor.le
        }

        joblib.dump(pipeline, filename)

        print(f"\n✓ Pipeline saved: {filename}")


# ============================
# MAIN TRAINING SCRIPT
# ============================

if __name__ == "__main__":

    from feature_extraction import FeatureExtractor

    print("Loading dataset...")

    df = pd.read_csv("jobs_cleaned.csv")

    print("Extracting features...")

    extractor = FeatureExtractor(df)
    X, y, _ = extractor.prepare_data()

    trainer = ModelTrainer(X, y, extractor)

    print("\nTraining models...")

    trainer.train_linear_regression()
    trainer.train_random_forest()
    trainer.train_gradient_boosting()

    # Save full pipeline (IMPORTANT)
    trainer.save_pipeline(
        "gradient_boosting",
        "data/models/salary_pipeline.pkl"
    )

    print("\n✓ All models trained and pipeline saved!")
    print("RUNNING FILE: src/model_training.py")