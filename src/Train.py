import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Özellik ve etiketleri yükle
X = np.load("../data/anjiyo_features.npy")
y = np.load("../data/anjiyo_labels.npy").flatten()

# --- Dengeleyici örnekleme (0 ve 1 sınıflarından eşit örnek) ---
idx_0 = np.where(y == 0)[0]
idx_1 = np.where(y == 1)[0]
np.random.seed(42)
if len(idx_1) < len(idx_0):
    idx_0 = np.random.choice(idx_0, size=len(idx_1), replace=False)
else:
    idx_1 = np.random.choice(idx_1, size=len(idx_0), replace=True)
idx_balanced = np.concatenate([idx_0, idx_1])
np.random.shuffle(idx_balanced)
X_bal = X[idx_balanced]
y_bal = y[idx_balanced]

# Eğitim ve doğrulama için veriyi ayır (dengeli örneklem)
X_train, X_val, y_train, y_val = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)

# --- Çapraz Doğrulama (Cross-Validation) ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("kNN için 5-fold CV sonuçları:")
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
cv_scores = cross_val_score(knn, X_bal, y_bal, cv=skf, scoring='accuracy', n_jobs=-1)
print("CV Doğruluk Ortalaması:", np.mean(cv_scores))
print("CV Doğruluk Skorları:", cv_scores)

print("LogisticRegression için 5-fold CV sonuçları:")
logreg = LogisticRegression(max_iter=200, solver='liblinear', class_weight='balanced')
cv_scores = cross_val_score(logreg, X_bal, y_bal, cv=skf, scoring='accuracy', n_jobs=-1)
print("CV Doğruluk Ortalaması:", np.mean(cv_scores))
print("CV Doğruluk Skorları:", cv_scores)

print("RandomForest için 5-fold CV sonuçları:")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
cv_scores = cross_val_score(rf, X_bal, y_bal, cv=skf, scoring='accuracy', n_jobs=-1)
print("CV Doğruluk Ortalaması:", np.mean(cv_scores))
print("CV Doğruluk Skorları:", cv_scores)

# k-NN sınıflandırıcı oluştur ve eğit
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn.fit(X_train, y_train)

# Doğrulama setinde performansı değerlendir
val_pred = knn.predict(X_val)
print("Doğrulama Doğruluğu (kNN):", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))
cm_knn = confusion_matrix(y_val, val_pred)
print("Confusion Matrix (kNN):\n", cm_knn)
plt.figure(figsize=(4,4))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (kNN)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('../evaluation/confusion_matrix_knn.png')
plt.close()

# Modeli kaydet
joblib.dump(knn, "../models/knn_model.joblib")
print("Model kaydedildi: ../models/knn_model.joblib")

# LogisticRegression (class_weight='balanced') ile eğitim
logreg = LogisticRegression(max_iter=200, solver='liblinear', class_weight='balanced')
logreg.fit(X_train, y_train)
val_pred = logreg.predict(X_val)
print("Doğrulama Doğruluğu (LogReg, balanced):", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))
cm_logreg = confusion_matrix(y_val, val_pred)
print("Confusion Matrix (LogReg):\n", cm_logreg)
plt.figure(figsize=(4,4))
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (LogReg)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('../evaluation/confusion_matrix_logreg.png')
plt.close()

# Modeli kaydet
joblib.dump(logreg, "../models/logreg_model.joblib")
print("Model kaydedildi: ../models/logreg_model.joblib")

# Random Forest ile eğitim
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)
val_pred = rf.predict(X_val)
print("Doğrulama Doğruluğu (RandomForest, balanced):", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))
cm_rf = confusion_matrix(y_val, val_pred)
print("Confusion Matrix (RandomForest):\n", cm_rf)
plt.figure(figsize=(4,4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (RandomForest)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('../evaluation/confusion_matrix_rf.png')
plt.close()

# Modeli kaydet
joblib.dump(rf, "../models/rf_model.joblib")
print("Model kaydedildi: ../models/rf_model.joblib")
