
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
from skimage import io as ioski
from skimage.color import rgb2gray
from skimage.transform import rotate, resize
import io
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'  # Modification du dossier de destination des images
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
model = None
model2 = None
model = load_model('Alphabet_Recognition')
model2 = load_model("model.h5")
alpha = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
amazigh_alpha = ['ⴰ', 'ⴱ', 'ⵛ', 'ⴷ', 'ⴹ', 'ⵄ', 'ⴼ', 'ⴳ', 'ⵖ', 'ⴳⵯ', 'ⵀ', 'ⵃ', 'ⵊ', 'ⴽ', 'ⴽⵯ',
                'ⵍ','ⵎ','ⵏ', 'ⵇ', 'ⵔ', 'ⵕ', 'ⵙ', 'ⵚ', 'ⵜ', 'ⵟ', 'ⵡ', 'ⵅ', 'ⵢ', 'ⵣ','ⵥ', 'ⴻ', 'ⵉ', 'ⵓ']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def thresholding(image,vall,img):
    
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,vall), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations = 1)
    (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h)
    img2 = img.copy()
    line_list = []

    for ctr in sorted_contours_lines:
        x,y,w,h = cv2.boundingRect(ctr)
        cv2.rectangle(img2, (x,y), (x+w, y+h), (40, 100, 250), 2)
        line_list.append([x, y, x+w, y+h])
    
    return line_list

def thresholding2(image,vall,img):
    
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray,120,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,vall), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations = 1)
    (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h)
    img2 = img.copy()
    line_list = []

    for ctr in sorted_contours_lines:
        x,y,w,h = cv2.boundingRect(ctr)
        cv2.rectangle(img2, (x,y), (x+w, y+h), (40, 100, 250), 2)
        line_list.append([x, y, x+w, y+h])
    
    return line_list

def alphabet_recognize(filepath):
    image = filepath
    blur_image=cv2.medianBlur(filepath,7)

    grey = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(grey ,150,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,5), np.uint8)
    dilated3 = cv2.dilate(thresh, kernel, iterations = 1)
    #plt.imshow(dilated3)
    contours,hierarchy= cv2.findContours(dilated3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    preprocessed_digits = []

    # initialize the reverse flag and sort index
    # handle if we need to sort in reverse
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    if contours and boundingBoxes:
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b:b[1][0], reverse=False))

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)

        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(blur_image, (x,y), (x+w, y+h), color=(255, 0, 0), thickness=2)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]

        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18,18))

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)

        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digits.append(padded_digit)
    alphabets=[]
    for digit in preprocessed_digits:
        prediction = model.predict(digit.reshape(1, 28, 28, 1) / 255., verbose=0)
        pred = alpha[np.argmax(prediction)]
        alphabets.append(pred)
    recognized_alphabets = ''.join(alphabets)
    #print(recognized_alphabets)
    return recognized_alphabets
    
def alphabet_recognize2(filepath):
    image = filepath
    blur_image=cv2.medianBlur(filepath,7)
    #plt.imshow(image)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(grey ,135,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,5), np.uint8)
    dilated3 = cv2.dilate(thresh, kernel, iterations = 1)
    
    contours,hierarchy= cv2.findContours(dilated3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preprocessed_digits = []

    # initialize the reverse flag and sort index
    # handle if we need to sort in reverse
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    if contours and boundingBoxes:
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b:b[1][0], reverse=False))

    #(contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                    #key=lambda b:b[1][0], reverse=False))

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)

        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(blur_image, (x,y), (x+w, y+h), color=(255, 0, 0), thickness=2)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]

        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (64,64))

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)

        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digits.append(padded_digit)
    img_size = 64
    alphabets=[]
    for digit in preprocessed_digits:
        resized = cv2.resize(digit, (img_size, img_size))
        normalized = resized / 255.0 
        input_data = np.reshape(normalized, (1, img_size, img_size, 1))
        prediction = model2.predict(input_data)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = amazigh_alpha[predicted_class_index]
        alphabets.append(predicted_class_label)
    recognized_alphabets = ''.join(alphabets)
    #print(recognized_alphabets)
    return recognized_alphabets
    

def amine(path):    
    image = cv2.imread(path)
    img = cv2.imread(path)
    #plt.imshow(image);
    #Line
    text = []
    lines = thresholding(image,200,image)
    for i,y in enumerate(lines):
        trtr = lines[i]
        roi_8=img[trtr[1]:trtr[3], trtr[0]:trtr[2]]
        #Word
        words = thresholding(roi_8,80,roi_8)
        index = len(words) - 1
        for w,u in enumerate(words):
            trtrr = words[index]
            index -= 1
            roi_9=roi_8[trtrr[1]:trtrr[3], trtrr[0]:trtrr[2]]
            word = alphabet_recognize(roi_9)
            #word = word + " "
            text.append(word)
            text.append(" ")
        text.append("/n")
    return text

def amine2(path):    
    image = cv2.imread(path)
    img = cv2.imread(path)
    #plt.imshow(image);
    #Line
    text = []
    lines = thresholding2(image,400,image)
    for i,y in enumerate(lines):
        trtr = lines[i]
        roi_8=img[trtr[1]:trtr[3], trtr[0]:trtr[2]]
        #Word
        words = thresholding2(roi_8,50,roi_8)
        index = len(words) - 1
        for w,u in enumerate(words):
            trtrr = words[index]
            index -= 1
            roi_9=roi_8[trtrr[1]:trtrr[3], trtrr[0]:trtrr[2]]
            word = alphabet_recognize2(roi_9)
            #word = word + " "
            text.append(word)
            text.append(" ")
        text.append("/n")
    return text

def my_text(path):
    my_text = amine(path)
    result = ""
    for t in my_text:
        if t == "/n":
            result += "\n"
        else :
            result += t
    return result

def my_text2(path):
    my_text = amine2(path)
    result = ""
    for t in my_text:
        if t == "/n":
            result += "\n"
        else :
            result += t
    return result

# def load_trained_model():
#     global model
#     model = load_model('Alphabet_Recognition')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    option = request.form.get('option')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check the selected option
        selected_option = request.form.get('option')
        
        # Perform actions based on the selected option
        if selected_option == 'English':
            # Preprocess the uploaded image
            image = cv2.imread(filepath)

            # Pass the correct file path to the amine function
            result = my_text(filepath)

            return render_template('index.html', text=result, filename=filename, selected_option=option)
        elif selected_option == 'Tifinagh':
            # Preprocess the uploaded image
            image = cv2.imread(filepath)

            # Pass the correct file path to the amine function
            result = my_text2(filepath)

            return render_template('index.html', text=result, filename=filename, selected_option=option)
        elif selected_option == 'Arabic':
            # Code for Option 3
            pass
        
        # Default behavior if no option is selected or option not recognized
        return render_template('index.html', filename=filename)
    else:
        return render_template('index.html', selected_option=option)



@app.route('/result')
def result():
    text = request.args.get('text')
    return render_template('result.html', text=text)

if __name__ == '__main__':
    # load_trained_model()
    app.run(debug=True)
