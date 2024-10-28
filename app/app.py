from http import HTTPStatus
import pandas as pd
from flask import request, Flask

from steps.predictor import Predictor
from utils.get_age_from_idade import get_age_from_idade

app = Flask(__name__)
predictor = Predictor()
model = predictor.load_model()

@app.route('/feature-importances', methods=['GET'])
def get_feature_importances():
    return dict(predictor.evaluate_importances())

@app.route('/predict', methods=['POST'])
def predict():
    request_body = request.get_json(silent=True)
    if not request_body:
        return {
            'status': HTTPStatus.BAD_REQUEST.name,
            'message': "Missing request body",
        }, 400

    data = {
        # Symptoms
        'FEBRE': int(request_body.get('febre', False)),
        'MIALGIA': int(request_body.get('mialgia', False)),
        'CEFALEIA': int(request_body.get('cefaleia', False)),
        'EXANTEMA': int(request_body.get('exantema', False)),
        'NAUSEA': int(request_body.get('nausea', False)),
        'DOR_COSTAS': int(request_body.get('dorCostas', False)),
        'ARTRALGIA': int(request_body.get('artralgia', False)),
        'PETEQUIA_N': int(request_body.get('petequia', False)),
        'DOR_RETRO': int(request_body.get('dorRetro', False)),
        # Other features
        'AGE': get_age_from_idade(request_body.get('idade')),
        'SYMPTOM_ONSET_MONTH': pd.to_datetime(request_body.get('dataDiagnosticoSintoma'), dayfirst=True).month,
    }
    df = pd.DataFrame([data])

    predicted_class = model.predict(df)
    probability = model.predict_proba(df)

    return {
        'predictedClass': int(predicted_class[0]),
        'classZeroProbability': probability[0][0],
        'classOneProbability': probability[0][1],
    }
