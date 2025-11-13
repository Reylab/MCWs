from bing_image_downloader import downloader
import cv2
from PIL import Image
import os

def download_images(names):

    # Create a list of arguments
    arguments = [{"keywords": name, "limit": 2, "print_urls": True} for name in names]

    # Download images for each name
    for name in names:
        downloader.download(name, limit=5, output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)


def crop_face(image_path):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Crop the face
        face = img[y:y+h, x:x+w]
        face_img = Image.fromarray(face)
        face_img = face_img.resize((160, 160))  # Resize the face to 160x160
        face_img.save(image_path)  # Save the cropped image



if __name__ == "__main__":
    # List of names
    names = ["obama", "prince", "deadpool"]

    download_images(names)

    # Crop the faces, doesn't work yet.
    for dirpath, dirnames, filenames in os.walk("dataset"):
        for filename in filenames:
            if filename.endswith(".jpg"):
                crop_face(os.path.join(dirpath, filename))
