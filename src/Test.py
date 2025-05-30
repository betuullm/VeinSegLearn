from PIL import Image
import numpy as np
import joblib
from Features import extract_features, load_image
from scipy.ndimage import binary_opening
import os

# Test görüntüsünü yükle
image_data = load_image("../data/anjiyo_test.jpg")

# Özellikleri çıkar
features = extract_features(image_data)

# Eğitilmiş modeli yükle (tüm özelliklerle eğitilmiş model)
model = joblib.load("../models/knn_model.joblib")

# Tahmin yap
pred_labels = model.predict(features)

# Maskeyi orijinal boyuta döndür
mask = pred_labels.reshape(image_data.shape[0], image_data.shape[1])

# Morfolojik açma işlemi (gürültü temizliği)
mask_clean = binary_opening(mask, structure=np.ones((3,3)))

# Sonucu kaydet
Image.fromarray((mask_clean * 255).astype(np.uint8)).save("../results/anjiyo_test_mask.png")
print("Test maskesi kaydedildi: ../results/anjiyo_test_mask.png")

# --- IoU Hesaplama (Eğer ground truth varsa) ---
label_path = "../data/anjiyo_test_label.bmp"
if os.path.exists(label_path):
    gt_mask = np.array(Image.open(label_path))
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[..., 0]  # Eğer renkli ise tek kanala indir
    gt_mask = (gt_mask > 127).astype(np.uint8)  # Binarize et
    pred_mask = mask_clean.astype(np.uint8)
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    iou = intersection / union if union > 0 else 0
    print(f"IoU (Intersection over Union): {iou:.4f}")
else:
    print("Uyarı: Test için ground truth maske bulunamadı, IoU hesaplanmadı.")
