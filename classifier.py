from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

if __name__ == '__main':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Provide path to the trained model")
    ap.add_argument("-i", "--image", required=True, help="Provide path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    print("Importing trained model...")
    model = load_model(args["model"])

    # classify the input image
    (nonFood, food) = model.predict(image)[0]

    label = "Food" if food > nonFood else "Non-Food"
    proba = food if food > nonFood else nonFood
    label = "{}: {:.2f}%".format(label, proba * 100)

    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)
