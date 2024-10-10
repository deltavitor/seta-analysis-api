import os
import joblib
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Predictor:

    def __init__(self):
        with open('config.yaml') as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        self.features = self.config['model']['features']
        self.store_path = self.config['model']['store_path']
        self.model = self.load_model()

    def evaluate_model(self, X_test, y_test):

        prediction = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction)
        recall = recall_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)

        print(f"[predict] Accuracy {accuracy}")
        print(f"[predict] Precision {precision}")
        print(f"[predict] Recall {recall}")
        print(f"[predict] F1 {f1}")

        return accuracy, precision, recall, f1

    def evaluate_importances(self):
        importances = list(self.model.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(self.features, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

        return feature_importances

    def load_model(self):
        store_path = os.path.join(self.store_path, 'model.pkl')
        return joblib.load(store_path)