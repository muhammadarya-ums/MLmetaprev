import joblib
import pandas as pd

# Nama fitur harus sama persis seperti saat training
FEATURES = ["IMT","LP","TD SISTOLIK","TD DIASTOLIK","GDA","HDL","Trigliserida"]

# load model sekali saat module diimpor
_model = joblib.load("model_rf.pkl")

def _to_float(x):
    """Bersihkan input: ganti koma -> titik, strip spasi, ubah ke float."""
    if x is None:
        return None
    s = str(x).strip()
    s = s.replace(",", ".")
    try:
        return float(s)
    except:
        raise ValueError(f"Nilai numerik tidak valid: {x}")

def predict_from_dict(input_dict):
    """
    input_dict: dict berisi fitur (bisa string dari form)
    contoh:
    {"IMT":"23.4", "LP":"85", "TD SISTOLIK":"120", "TD DIASTOLIK":"80", "GDA":"95", "HDL":"55", "Trigliserida":"120"}
    mengembalikan dict: {"label": "...", "proba": {"Bebas ...":0.6, ...}}
    """
    # Pastikan semua fitur ada
    row = []
    for f in FEATURES:
        if f not in input_dict:
            raise KeyError(f"Fitur hilang: {f}")
        row.append(_to_float(input_dict[f]))

    df = pd.DataFrame([row], columns=FEATURES)
    pred = _model.predict(df)[0]
    pred_diag = pred.split(",")[0]

    proba = _model.predict_proba(df)[0]
    classes = list(_model.classes_)
    proba_dict = {cls: float(prob) for cls, prob in zip(classes, proba)}

    return {"label": pred}
