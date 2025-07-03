# =============================================================================
# Test File for Paddle OCR 02/07/2025
# =============================================================================

from paddleocr import PaddleOCR
import cv2
from matplotlib import pyplot as plt
import fitz
import numpy as np
import os.path
import tkinter as tk
from tkinter import filedialog

def pngDetector(image_path) :

    ocr = PaddleOCR(use_angle_cls=True, lang="fr")
    img = cv2.imread(image_path)

    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
    plt.show()

    result = ocr.predict(img)

    # Print all recognized texts
    for text in result[0]['rec_texts']:
        print(text)

def pdfDetector(pdf_path) : 


    ocr = PaddleOCR(use_angle_cls=True, lang="fr")

    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    plt.figure(figsize=(10, 12))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Aperçu de la première page")
    plt.show()

    result = ocr.predict(img)

    for line in result[0]['rec_texts']:
        print(line)


def pngOrPdf(filepath) : 
    ext = os.path.splitext(filepath)[1]
    print(ext)
    if ext == ".pdf" : 
        pdfDetector(filepath)
        
    elif ext == ".png" : 
        pngDetector(filepath)
    
    else:
        raise NameError("L'extension du fichier n'est pas supporté par le programme")
    


def chooseFile():
    root = tk.Tk()
    root.withdraw()
    chemin = filedialog.askopenfilename(title="Choisissez un fichier")
    if chemin:
        nom_fichier = os.path.basename(chemin)
        print(f"Fichier sélectionné : {nom_fichier}")
        return nom_fichier
    else:
        print("Aucun fichier sélectionné.")
        return None
        
        
filepath = chooseFile()

pngOrPdf(filepath)
        
        
        
        