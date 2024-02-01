from flask import Flask, request, render_template
from datetime import datetime
from fuzzywuzzy import fuzz
import face_recognition
import cv2
import os
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'UPLOAD_FOLDER'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def form_post():
    Name_Lost = request.form['left_name']
    Name_Found = request.form['right_name']
    Location_Lost = request.form['left_location']
    Location_Found = request.form['right_location']
    Features_Lost = request.form['left_facial_features']
    Features_Found = request.form['right_facial_features']
    Keywords_Lost = request.form['left_keywords']
    Keywords_Found = request.form['right_keywords']
    Age_Lost = int(request.form['left_age'])
    Age_Found = int(request.form['right_age'])
    Gender_Lost = request.form['left_gender']
    Gender_Found = request.form['right_gender']
    Date_Lost = datetime.strptime(request.form['left_date'], '%Y-%m-%d')
    Date_Found = datetime.strptime(request.form['right_date'], '%Y-%m-%d')
    left_image = request.files['left_image']
    right_image = request.files['right_image']

    left_image.save(os.path.join(app.config['UPLOAD_FOLDER'], "left.png"))
    right_image.save(os.path.join(app.config['UPLOAD_FOLDER'], "right.png"))

    difference_date = (Date_Found - Date_Lost).days
    difference_age = abs(Age_Found - Age_Lost)

    gender_ratio = 100 if Gender_Lost == Gender_Found else 0
    age_ratio = 100 if difference_age <= 3 else 90 if difference_age <= 5 else 70 if difference_age <= 7 else 0
    date_ratio = 100 if difference_date < 10 else 90 if difference_date < 20 else 80 if difference_date < 30 else 70 if difference_date < 60 else 60 if difference_date < 90 else 30

    name_ratio = fuzz.ratio(Name_Lost, Name_Found)
    location_ratio = fuzz.ratio(Location_Lost, Location_Found)
    features_ratio = fuzz.ratio(Features_Lost, Features_Found)
    keywords_ratio = fuzz.ratio(Keywords_Lost, Keywords_Found)

    result = f"Name Match: {name_ratio}% | \nLocation Match: {location_ratio}% | \nFeatures Match: {features_ratio}% | \nKeywords Match: {keywords_ratio}% | \nDate Match: {date_ratio}% | \nAge Match: {age_ratio}% | \nGender Match: {gender_ratio}%"
    print(result)
    
    def draw_landmarks(image, face_landmarks):
        for facial_feature in face_landmarks.keys():
            points = face_landmarks[facial_feature]
            for i in range(len(points) - 1):
                cv2.line(image, points[i], points[i+1], (0, 255, 0), 2)
            cv2.line(image, points[-1], points[0], (0, 255, 0), 2)

    def process_image(image_path, target_height=600):
        image = face_recognition.load_image_file(image_path)
        face_landmarks_list = face_recognition.face_landmarks(image)
        
        image_cv2 = cv2.imread(image_path)

        for face_landmarks in face_landmarks_list:
            draw_landmarks(image_cv2, face_landmarks)

        current_height = image_cv2.shape[0]
        scale_factor = target_height / current_height
        image_cv2_resized = cv2.resize(image_cv2, (0, 0), fx=scale_factor, fy=scale_factor)

        return image_cv2_resized

    image_paths = ["UPLOAD_FOLDER/left.png", "UPLOAD_FOLDER/right.png"]

    def find_face_encodings(image_path):
        image = cv2.imread(image_path)
        face_enc = face_recognition.face_encodings(image)
        return face_enc[0]

    image_1 = find_face_encodings("UPLOAD_FOLDER/left.png")
    image_2  = find_face_encodings("UPLOAD_FOLDER/right.png")



    distance = face_recognition.face_distance([image_1], image_2)
    distance = round(distance[0] * 100)
    accuracy = 136-round(distance)
    print(f"[{accuracy}% Match]✅")

    images = [process_image(image_path) for image_path in image_paths]
    concatenated_image = np.concatenate(images, axis=1)

    text_to_display = f"{str(accuracy)}% Match"
    cv2.putText(concatenated_image, text_to_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 73), 2, cv2.LINE_AA)

    cv2.putText(concatenated_image, "Resc-U | AI for Good", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 73), 2, cv2.LINE_AA)


    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Save the image with error handling
    output_path = os.path.join(static_dir, 'concatenated_image.jpg')
    try:
        cv2.imwrite(output_path, concatenated_image)
        print(f"Image saved successfully at {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
    return render_template('index.html', result=result, image_file='static/concatenated_image.jpg')

    

if __name__ == '__main__':
    app.run(debug=True)
