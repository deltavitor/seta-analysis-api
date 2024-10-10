import os
import joblib
import yaml
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class Trainer:

    def __init__(self):
        with open('config.yaml') as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        self.features = self.config['model']['features']
        self.target = self.config['model']['target']
        self.random_state = self.config['random_state']
        self.test_size = self.config['training']['test_size']
        self.parameters = self.config['model']['parameters']
        self.store_path = self.config['model']['store_path']

    def train_model(self, X_train, y_train):
        clf = RandomForestClassifier(random_state=self.random_state, **self.parameters)
        clf.fit(X_train, y_train)

        return clf

    def save_model(self, model):
        store_path = os.path.join(self.store_path, 'model.pkl')
        joblib.dump(model, store_path)

    def split_train_test_data(self, data):
        X = data[self.features]
        y = data[self.target]
        X, y = shuffle(X, y, random_state=self.random_state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state, test_size=self.test_size)
        return X_train, X_test, y_train, y_test

    def split_features_target(self, data):
        X = data[self.features]
        y = data[self.target]
        return X, y