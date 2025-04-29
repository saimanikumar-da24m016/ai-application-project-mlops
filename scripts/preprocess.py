# scripts/preprocess.py
import os
from pathlib import Path
from PIL import Image

def resize_images(input_dir: str, output_dir: str, size=(224, 224)):
    """
    Walk through input_dir/{class}/images, resize each to `size`,
    and save into output_dir/<class> keeping filenames.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for cls in os.listdir(input_dir):
        in_cls = Path(input_dir) / cls
        out_cls = Path(output_dir) / cls
        out_cls.mkdir(parents=True, exist_ok=True)
        for img_file in in_cls.glob("*.[jJ][pP][gG]"):
            img = Image.open(img_file).convert("RGB")
            img = img.resize(size)
            img.save(out_cls / img_file.name)
