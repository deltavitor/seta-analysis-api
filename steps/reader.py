from dbfread import DBF
import pandas as pd
import yaml

class Reader:

    def __init__(self):
        with open('config.yaml') as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        self.train_path = self.config['data']['train_path']

    def read_data(self):
        print("[read] Reading dataset")
        dbf = DBF(self.train_path, ignore_missing_memofile=True, char_decode_errors='ignore')
        data = pd.DataFrame(iter(dbf))
        return data