from PIL import Image
import numpy as np

def load_bmp_image(file_path):
    try:
        # BMP dosyasını aç
        image = Image.open(file_path)
        
        # Görüntüyü numpy dizisine çevir
        image_array = np.array(image)
        
        return image_array
    except Exception as e:
        print(f"Hata: {str(e)}")
        return None

def create_label_matrix(image_array):
    # Görüntü gri seviye değilse, gri seviyeye çevir
    if len(image_array.shape) > 2:
        gray_image = np.mean(image_array, axis=2)
    else:
        gray_image = image_array
    
    # Görüntüyü 1 boyuta indir (262144x1)
    flat_gray = gray_image.flatten().reshape(-1, 1)
    print("Min:", flat_gray.min(), "Max:", flat_gray.max())  # Piksel değerlerini kontrol et
    # Binary etiketleme: beyaz (1) -> 1, siyah (0) -> 0
    label_matrix = (flat_gray > 0.5).astype(np.uint8)
    return label_matrix

# Görüntüyü yükle
image_data = load_bmp_image("../data/anjiyo_label.bmp")

if image_data is not None:
    # Etiket matrisini oluştur
    labels = create_label_matrix(image_data)
    
    print(f"Görüntü boyutu: {image_data.shape}")
    print(f"Etiket matrisi boyutu: {labels.shape}")
    print(f"Beyaz piksel sayısı (1): {np.sum(labels == 1)}")
    print(f"Siyah piksel sayısı (0): {np.sum(labels == 0)}")
    
    # Etiket matrisini kaydet (isteğe bağlı)
    np.save("data/anjiyo_labels.npy", labels)