import joblib

def predict(model, data):
    loaded_model = joblib.load(model)
    return loaded_model.predict([data])[0]