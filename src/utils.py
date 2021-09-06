from PIL import Image
from io import BytesIO

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image