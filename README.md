# VeinClassifier

Bu proje, medikal görüntülerde damar segmentasyonu ve sınıflandırması için temel bir makine öğrenimi pipeline'ı sunar.

## Klasör Yapısı

```
VeinClassifier/
├── data/               # (Veri burada tutulmaz, .gitignore içinde tutulur)
├── src/                # Ana kodlar (özellik çıkarımı, model eğitimi vs.)
│   ├── feature_extraction.py
│   ├── train_classifier.py
│   ├── predict.py
│   └── utils.py
├── notebooks/          # Jupyter analizleri veya testler
│   └── vein_classification_demo.ipynb
├── results/            # Sonuç görselleri veya çıktı dosyaları
├── requirements.txt    # Kullanılan kütüphaneler
├── .gitignore
└── README.md
```

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
2. `data/` klasörüne eğitim ve test görüntülerinizi ekleyin (bu klasör git ile takip edilmez).

## Kullanım

- Özellik çıkarımı: `src/feature_extraction.py`
- Model eğitimi: `src/train_classifier.py`
- Tahmin: `src/predict.py`
- Analiz ve görselleştirme: `notebooks/vein_classification_demo.ipynb`

## Notlar
- `data/` ve büyük veri dosyaları `.gitignore` ile hariç tutulur.
- Sonuçlar ve çıktı görselleri `results/` klasöründe saklanır.
