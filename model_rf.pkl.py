import joblib

# model = RandomForestClassifier(...)   # model sudah dilatih
joblib.dump(model, "model_rf.pkl")
print("Model tersimpan: model_rf.pkl")
