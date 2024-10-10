from steps.predictor import Predictor
from steps.reader import Reader
from steps.sanitizer import Sanitizer
from steps.trainer import Trainer

def main():
    reader = Reader()
    data = reader.read_data()

    sanitizer = Sanitizer()
    data = sanitizer.sanitize_data(data)
    data = sanitizer.map_new_columns(data)
    data = sanitizer.cleanup_unused_columns(data)

    trainer = Trainer()
    X_train, X_test, y_train, y_test = trainer.split_train_test_data(data)
    clf = trainer.train_model(X_train, y_train.values.ravel())
    trainer.save_model(clf)

    predictor = Predictor()
    predictor.evaluate_model(X_test, y_test)
    predictor.evaluate_importances()

if __name__ == '__main__':
    main()
