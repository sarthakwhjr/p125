import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
X=np.load("image.npz")["arr_0"]
y=pd.read_csv("labels.csv")["labels"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0  
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)
def getpredict(img):
    openimg=Image.open(img)
    imgbw=openimg.convert("L")
    imgresize=imgbw.resize((28,28),Image.ANTIALIAS)
    pixel=20
    minpix=np.percentile(imgresize,pixel)
    pixcrop=np.clip(imgresize-minpix,0,255)
    maxpix=np.max(imgresize)
    pixcrop=np.asarray(pixcrop)/maxpix
    testsample=np.array(pixcrop).reshape(1,784)
    testpredict=clf.predict(testsample)
    return testpredict[0]