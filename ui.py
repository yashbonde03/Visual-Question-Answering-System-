from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import io
from io import BytesIO
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'


@app.route('/')
def home():
    message = "11"
    #return render_template('index.html', statusCode='Success', message=message)
    return render_template('front_page.html', statusCode='Success', message=message)

@app.route('/about')
def about():
    message = "11"
    #return render_template('index.html', statusCode='Success', message=message)
    return render_template('about.html', statusCode='Success', message=message)

@app.route('/predict')
def predict():
    message = "11"
    #return render_template('index.html', statusCode='Success', message=message)
    return render_template('index.html', statusCode='Success', message=message)

@app.route('/intro')
def intro():
    message = "11"
    return render_template('intro.html', statusCode='Success', message=message)

@app.route('/index', methods=['POST', 'GET'])
def upload():
    # Get the uploaded image from the request
    file = request.files['image']
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Load uploaded image using cv2.imread()
    print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(
        img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    with open('coco-labels', 'r') as f:
        classes = f.read().splitlines()
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    # New Code

    # Detecting objects
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    # to show information on the screen
    boxes = []
    confidences = []
    class_ids = []
    import numpy as np
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    cnt = 0
    label1 = ""
    color_name1 = ""
    mainObj = ""
    maincnt = 0
    obj=""
    pred=""
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        # label1 = label
        obj=label
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label + " " + confidence,
                    (x, y-5), font, 1, (255, 255, 255), 1)
        cnt = cnt+1

    print(cnt)

    cv2.imshow('Output_Image', img)
    # cv2.imwrite('test_image_32_result.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('checking Values')
    checkValue = request.form['p']
    print(checkValue)
    if checkValue == "color":


        # Define a function to get the name of the closest color in the webcolors database
        def get_color_name(rgb):
            min_colors = {}
            import webcolors
            for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(key)
                rd = (r_c - rgb[0]) ** 2
                gd = (g_c - rgb[1]) ** 2
                bd = (b_c - rgb[2]) ** 2
                min_colors[(rd + gd + bd)] = name
            return min_colors[min(min_colors.keys())]

            # Get the dominant color of the img
        import numpy as np
        pixels = np.float32(img.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]

        # Get the name of the closest color in the webcolors database
        color_name = get_color_name(dominant)
        color_name1 = color_name
        # Show the dominant color and its name
        # print(color_name)
        text = f"RGB({int(dominant[0])}, {int(dominant[1])}, {int(dominant[2])}) - {color_name}"
        print(
            f"Dominant color: RGB({int(dominant[0])}, {int(dominant[1])}, {int(dominant[2])}) - {color_name}")
        # print(text)

    elif checkValue == "object":
        def get_color_name(rgb):
            min_colors = {}
            import webcolors
            for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(key)
                rd = (r_c - rgb[0]) ** 2
                gd = (g_c - rgb[1]) ** 2
                bd = (b_c - rgb[2]) ** 2
                min_colors[(rd + gd + bd)] = name
            return min_colors[min(min_colors.keys())]
        import numpy as np
        pixels = np.float32(img.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]

        # Get the name of the closest color in the webcolors database
        color_name = get_color_name(dominant)

        # Show the dominant color and its name
        print(color_name)
        text = f"RGB({int(dominant[0])}, {int(dominant[1])}, {int(dominant[2])}) - {color_name}"
        print(
            f"Dominant color: RGB({int(dominant[0])}, {int(dominant[1])}, {int(dominant[2])}) - {color_name}")
        print(text)

        mainObj = label
        print('mainObj')
        print(mainObj)

    elif checkValue == 'objCount':


        
        def get_color_name(rgb):
            min_colors = {}
            import webcolors
            for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(key)
                rd = (r_c - rgb[0]) ** 2
                gd = (g_c - rgb[1]) ** 2
                bd = (b_c - rgb[2]) ** 2
                min_colors[(rd + gd + bd)] = name
            return min_colors[min(min_colors.keys())]
        import numpy as np
        pixels = np.float32(img.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]

        # Get the name of the closest color in the webcolors database
        color_name = get_color_name(dominant)

        # Show the dominant color and its name
        print(color_name)
        text = f"RGB({int(dominant[0])}, {int(dominant[1])}, {int(dominant[2])}) - {color_name}"
        print(
            f"Dominant color: RGB({int(dominant[0])}, {int(dominant[1])}, {int(dominant[2])}) - {color_name}")
        print(text)
        maincnt = cnt

    elif checkValue == "txt_name":
        def get_color_name(rgb):
            min_colors = {}
            import webcolors
            for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(key)
                rd = (r_c - rgb[0]) ** 2
                gd = (g_c - rgb[1]) ** 2
                bd = (b_c - rgb[2]) ** 2
                min_colors[(rd + gd + bd)] = name
            return min_colors[min(min_colors.keys())]
        import numpy as np
        pixels = np.float32(img.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]

        # Get the name of the closest color in the webcolors database
        color_name = get_color_name(dominant)

        # Show the dominant color and its name
        print(color_name)
        text = f"RGB({int(dominant[0])}, {int(dominant[1])}, {int(dominant[2])}) - {color_name}"
        print(
            f"Dominant color: RGB({int(dominant[0])}, {int(dominant[1])}, {int(dominant[2])}) - {color_name}")
        print(text)

        enteredValue=request.form['txtName']
        if enteredValue==obj:
            pred="TRUE"
        else:
            pred="FALSE"


    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    

    return render_template('index.html',  message1=color_name1, up_img=path, obj=mainObj, count=maincnt,pred1=pred)


if __name__ == '__main__':
    app.run(debug=True)
