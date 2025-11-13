
import cv2
from PIL import Image
import os
from pathlib import Path
import shutil
import sys
sys.path.append(os.path.dirname(__file__))
from bing_downloader import BingDownloader

class TaskImages:
    """
    TaskImages class
    """
    def __init__(self):
        self.task_images_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'task_images'))

    def download_images(self, output_dir, names, filter="", limit=5, force_replace=False, verbose=True):
        """
        Download images from Bing Image Search

        Args:
        	names: List of names
        """
        # Download images for each name
        for name in names:
            self.download(name, limit=limit, output_dir=output_dir, filter=filter,
                          adult_filter_off=True, force_replace=force_replace, timeout=60, verbose=verbose)

    def download(self, concept, related_concept="", name="", limit=100, output_dir='dataset', adult_filter_off=True, 
                force_replace=False, timeout=10, filter=filter, verbose=True):

        # engine = 'bing'
        if adult_filter_off:
            adult = 'off'
        else:
            adult = 'on'

        if related_concept:
            query = related_concept + " " + concept
        elif name:
            query = name + " " + concept
        else:
            query = concept

        

        if force_replace:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        # else:
        #     if os.path.exists(image_dir):
        #         print(f"[Error] {image_dir} already exists. Use force_replace=True to overwrite.")
        #         return


        # check directory and create if necessary
        try:
            os.makedirs(output_dir, exist_ok=True)

        except Exception as e:
            print('[Error]Failed to create directory.', e)
            
        print(f"Downloading Images to {output_dir}")
        bing = BingDownloader(query=query, concept=concept, related_concept=related_concept, name=name,
                              limit=limit, output_dir=output_dir, 
                              adult=adult, timeout=timeout, filter=filter, verbose=verbose)
        bing.run()
    
    def crop_face(self, image_path):
        """
        Crop the face from an image

        Args:
        	image_path: Path to the image
        """
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
    names = ["obama", "prince", "marquette"]
    task = TaskImages()
    task.download_images(names, force_replace=True, verbose=True)
    # Crop the faces, doesn't work yet.
        # for dirpath, dirnames, filenames in os.walk(self.task_images_path):
        #     for filename in filenames:
        #         if filename.endswith(".jpg"):
        #             self.crop_face(os.path.join(dirpath, filename))
