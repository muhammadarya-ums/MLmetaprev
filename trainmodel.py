import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ==== 1. Load dataset ====
df = pd.read_excel("dataset fix.xlsx")  # Pastikan file ada di folder yang sama

# Kolom fitur sesuai predict_model.py
FEATURES = ["IMT", "LP", "TD SISTOLIK", "TD DIASTOLIK", "GDA", "HDL", "Trigliserida"]
LABEL = "Label"  # Nama kolom label di dataset

# ==== 2. Fungsi bersihkan data ====
def to_float(x):
    if isinstance(x, str):
        x = x.replace(",", ".").strip()  # Ganti koma ke titik dan hapus spasi
    try:
        return float(x)
    except:
        return None

# Bersihkan semua fitur numerik
for col in FEATURES:
    df[col] = df[col].apply(to_float)

# Hapus baris yang masih ada nilai kosong
df = df.dropna(subset=FEATURES + [LABEL])

# ==== 3. Pisahkan fitur & label ====
X = df[FEATURES]
y = df[LABEL]

# ==== 4. Split data train-test ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==== 5. Buat & latih model ====
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# ==== 6. Evaluasi ====
y_pred = model.predict(X_test)
print("\n=== Laporan Klasifikasi ===")
print(classification_report(y_test, y_pred))

# ==== 7. Simpan model ====
joblib.dump(model, "model_rf.pkl")
print("\n✅ Model tersimpan sebagai model_rf.pkl")
