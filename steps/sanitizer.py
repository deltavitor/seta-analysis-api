import pandas as pd
import yaml

from utils.get_age_from_idade import get_age_from_idade

class Sanitizer:

    def __init__(self):
        with open("config.yaml") as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        self.symptoms = self.config['model']['symptoms']
        self.features = self.config['model']['features']
        self.target = self.config['model']['target']
        self.processing_columns = self.config['model']['processing_columns']
        self.target_years = self.config['model']['target_years']

    def sanitize_data(self, data):
        print(f"[sanitize] Initial data length: {len(data)}")

        data.drop(data.columns.difference(self.features + self.target + self.processing_columns), axis=1, inplace=True)
        print(f"[sanitize] Removed unnecessary columns")

        data.dropna(inplace=True)
        print(f"[sanitize] Removed blank values - New size: {len(data)}")

        target_map = {'10': 1, '11': 1, '12': 1, '5': 0, '13': 0, '8': 0}
        data[self.target] = data[self.target].apply(lambda x: x.map(target_map))
        print("[sanitize] Mapped target values")

        symptoms_map = {'1': 1, '2': 0}
        data[self.symptoms] = data[self.symptoms].apply(lambda x: x.map(symptoms_map))
        print("[sanitize] Mapped symptoms to binary")

        numeric_columns = ['CRITERIO', 'RESUL_SORO', 'RESUL_NS1']
        data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        print("[sanitize] Mapped other numeric columns")

        data['DT_SIN_PRI'] = pd.to_datetime(data['DT_SIN_PRI'], format='%d/%m/%y', dayfirst=True, errors='coerce')
        data.dropna(subset=['DT_SIN_PRI'], inplace=True)
        print(f"[sanitize] Mapped symptom onset dates and dropped invalid dates - New size: {len(data)}")

        data = data[data['DT_SIN_PRI'].dt.year.isin(self.target_years)]
        print(f"[sanitize] Removed out of bounds symptom dates - New size: {len(data)}")

        condition_valid_age = ((data['NU_IDADE_N'] > 1000) & (data['NU_IDADE_N'] < 4999))
        data = data[condition_valid_age]
        print(f"[sanitize] Applied condition: Valid age required - New size: {len(data)}")

        condition_classi_fin_1 = (data['CLASSI_FIN'] == 1) & ((data['RESUL_SORO'] == 1) | (data['RESUL_NS1'] == 1))
        condition_classi_fin_0 = (data['CLASSI_FIN'] == 0) & (data['RESUL_SORO'] == 2)
        data = data[condition_classi_fin_1 | condition_classi_fin_0]
        print(f"[sanitize] Applied condition: Valid lab test required - New size: {len(data)}")

        data = data[~(data[self.symptoms] == 0).all(axis=1)]
        print(f"[sanitize] Applied condition: At least one symptom required - New size: {len(data)}")

        return data

    def map_new_columns(self, data):
        data['SYMPTOM_ONSET_MONTH'] = data['DT_SIN_PRI'].dt.month
        data['AGE'] = data['NU_IDADE_N'].apply(get_age_from_idade)
        print("[sanitize] Created new columns")

        return data

    def cleanup_unused_columns(self, data):
        data.drop(data.columns.difference(self.features + self.target), axis=1, inplace=True)
        print("[sanitize] Removed columns other than features and target")

        return data
