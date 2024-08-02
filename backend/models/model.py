import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error
import shap


class Model:

    def __init__(self):
        pass

    def train(self, X, y): # train our model
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.gbdt.fit(self.X_train, self.y_train)
        self.y_train_pred_gbdt = self.gbdt.predict(self.X_train)
        self.y_test_pred_gbdt = self.gbdt.predict(self.X_test)
        X_train_blend = np.hstack((self.X_train, self.y_train_pred_gbdt.reshape(-1, 1)))
        X_test_blend = np.hstack((self.X_test, self.y_test_pred_gbdt.reshape(-1, 1)))
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf.fit(X_train_blend, self.y_train)
        self.y_train_pred_rf = self.rf.predict(X_train_blend)
        self.y_test_pred_rf = self.rf.predict(X_test_blend)
        self.X_train_stack = np.vstack((self.y_train_pred_gbdt, self.y_train_pred_rf)).T
        self.X_test_stack = np.vstack((self.y_test_pred_gbdt, self.y_test_pred_rf)).T
        self.ridge = Ridge(alpha=1.0)
        self.ridge.fit(self.X_train_stack, self.y_train)
        self.y_train_pred_ridge = self.ridge.predict(self.X_train_stack)
        self.y_test_pred_ridge = self.ridge.predict(self.X_test_stack)

    def evaluate(self):  # record model performance
        metrics = {
            "Mean Squared Error": mean_squared_error,
            "Root Mean Squared Error": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            "Mean Absolute Error": mean_absolute_error,
            "R² Score": r2_score,
            "Explained Variance Score": explained_variance_score,
            "Median Absolute Error": median_absolute_error,
        }
        results = []
        results.append("Training Performance:")
        for name, metric in metrics.items():
            results.append(f"{name}: {metric(self.y_train, self.y_train_pred_ridge)}")
        results.append("Testing Performance:")
        for name, metric in metrics.items():
            results.append(f"{name}: {metric(self.y_test, self.y_test_pred_ridge)}")

        return results

    def interpret(self):  # explain model with plots
        explainer = shap.Explainer(self.ridge, self.X_test_stack)
        shap_values = explainer(self.X_test_stack)
        return shap_values
