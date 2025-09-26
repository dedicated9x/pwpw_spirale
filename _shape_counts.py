import os
import glob
from PIL import Image
import pandas as pd

# katalog ze zdjęciami
folder = "/home/admin2/Downloads/Shape detection/ShapeDetector raw"
# folder = "/home/admin2/Downloads/Shape detection/ShapeDetector treshold"

# znajdź wszystkie pliki .jpg (bez rozróżniania wielkości liter w rozszerzeniu)
files = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.JPG"))

sizes = []

for f in files:
    try:
        with Image.open(f) as img:
            sizes.append(img.size)  # (width, height)
    except Exception as e:
        print(f"Nie udało się otworzyć {f}: {e}")

# wrzucamy do DataFrame
df = pd.DataFrame(sizes, columns=["width", "height"])

# liczymy value_counts
counts = df.value_counts().reset_index(name="count")

print(counts)
