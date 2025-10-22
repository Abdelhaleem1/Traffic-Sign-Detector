from PIL import Image

def what(file, h=None):
    try:
        with Image.open(file) as img:
            return img.format.lower()
    except Exception:
        return None
