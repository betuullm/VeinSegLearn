from PIL import Image
import numpy as np
from scipy.ndimage import uniform_filter, generic_filter, laplace, sobel
from skimage.filters import gabor
from skimage.util import img_as_float
from skimage.measure import shannon_entropy
from skimage import exposure, filters
from skimage.morphology import white_tophat, disk

def load_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array

def local_entropy(image, size=3):
    # Lokal entropi (texture özelliği)
    def entropy_func(values):
        hist, _ = np.histogram(values, bins=8, range=(0, 1), density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    return generic_filter(image, entropy_func, size=size)

def local_max_eigenvalue(image, size=3):
    # Lokal pencere için kovaryans matrisinin en büyük özdeğeri
    def eig_func(values):
        vals = values.reshape((size, size))
        cov = np.cov(vals)
        eigvals = np.linalg.eigvalsh(cov)
        return np.max(eigvals)
    return generic_filter(image, eig_func, size=size)

def preprocess_image(image_array):
    # Gri seviye
    if len(image_array.shape) > 2:
        gray = np.mean(image_array, axis=2)
    else:
        gray = image_array.astype(np.float32)
    gray = img_as_float(gray)
    gray = np.clip(gray, 0, 1)
    # Top-hat transform (damar vurgulama)
    selem = disk(7)  # Damar kalınlığına göre ayarlanabilir
    gray_tophat = white_tophat(gray, selem)
    # Daha agresif kontrast (CLAHE)
    gray_eq = exposure.equalize_adapthist(gray_tophat, clip_limit=0.01)
    # Gaussian blur (gürültü azaltma)
    gray_blur = filters.gaussian(gray_eq, sigma=0.7)
    return gray_blur

def extract_features(image_array):
    # Ön işleme uygula
    gray = preprocess_image(image_array)
    # 3x3 komşuluk ortalaması
    mean3x3 = uniform_filter(gray, size=3)
    # 3x3 komşuluk standart sapması
    std3x3 = generic_filter(gray, np.std, size=3)
    # Laplacian
    lap = laplace(gray)
    # Sobel X ve Y
    sobel_x = sobel(gray, axis=0)
    sobel_y = sobel(gray, axis=1)
    # Gabor filtreleri (4 yön, 2 frekans)
    gabor_feats = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for freq in [0.3, 0.6]:
            real, imag = gabor(gray, frequency=freq, theta=theta)
            gabor_feats.append(real.flatten())
            gabor_feats.append(imag.flatten())
    # Distance to center
    h, w = gray.shape
    y, x = np.indices((h, w))
    center_y, center_x = h / 2, w / 2
    dist_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    dist_center = dist_center / dist_center.max()
    # Normalize koordinatlar
    x_norm = (x / (w - 1)).flatten()
    y_norm = (y / (h - 1)).flatten()
    # Lokal entropi
    entropy = local_entropy(gray, size=3)
    # Lokal pencere özdeğeri
    max_eig = local_max_eigenvalue(gray, size=3)
    # Özellik matrisini oluştur
    features = np.column_stack([
        gray.flatten(),
        mean3x3.flatten(),
        std3x3.flatten(),
        lap.flatten(),
        sobel_x.flatten(),
        sobel_y.flatten(),
        *gabor_feats,
        dist_center.flatten(),
        x_norm,
        y_norm,
        entropy.flatten(),
        max_eig.flatten()
    ])
    return features

# Görüntüyü yükle
image_data = load_image("../data/anjiyo.jpg")

# Özellikleri çıkar
features = extract_features(image_data)
print(f"Özellik matrisi boyutu: {features.shape}")
# İsteğe bağlı: Kaydet
np.save("../data/anjiyo_features.npy", features)
