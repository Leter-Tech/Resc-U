from flask import Flask, request, render_template
from datetime import datetime
from fuzzywuzzy import fuzz
import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import mediapipe as mp

app = Flask(__name__)

UPLOAD_FOLDER = 'UPLOAD_FOLDER'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def get_face_embedding(image_path):
    img = Image.open(image_path)
    face = mtcnn(img)
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0))
    return embedding

def calculate_similarity(embedding1, embedding2):
    cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return cos_sim.item()

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
    left_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "left.png")
    right_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "right.png")

    left_image = request.files['left_image']
    right_image = request.files['right_image']

    left_image.save(left_image_path)
    right_image.save(right_image_path)

    difference_date = (Date_Found - Date_Lost).days
    difference_age = abs(Age_Found - Age_Lost)

    gender_ratio = (100 if Gender_Lost == Gender_Found else 0) * 6/100
    age_ratio = (100 if difference_age <= 3 else 90 if difference_age <= 5 else 70 if difference_age <= 7 else 10) * 2/100
    date_ratio = (100 if difference_date < 10 else 90 if difference_date < 20 else 80 if difference_date < 30 else 70 if difference_date < 60 else 60 if difference_date < 90 else 10) * 3/100
    name_ratio = fuzz.ratio(Name_Lost, Name_Found) * 7/100
    location_ratio = fuzz.ratio(Location_Lost, Location_Found) * 6/100
    features_ratio = fuzz.ratio(Features_Lost, Features_Found) * 9/100
    keywords_ratio = fuzz.ratio(Keywords_Lost, Keywords_Found) * 6/100
    
    embedding1 = get_face_embedding(left_image_path)
    embedding2 = get_face_embedding(right_image_path)

    similarity = calculate_similarity(embedding1, embedding2) if embedding1 is not None and embedding2 is not None else 0
    optimization = round(similarity*100) + 20
    if optimization > 100:
        optimization = 99
    face_ratio = optimization*65/100
    final_match_percentage = gender_ratio + age_ratio + date_ratio + name_ratio + location_ratio + features_ratio + keywords_ratio + face_ratio
    final_match_percentage = round(final_match_percentage)    
    print(f"[{final_match_percentage:.2f}% Match]")
    
    def process_image(image_path, target_height=600):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image_rgb, 
                    landmark_list=face_landmarks, 
                    connections=mp_face_mesh.FACEMESH_TESSELATION, 
                    landmark_drawing_spec=drawing_spec, 
                    connection_drawing_spec=drawing_spec)
        
        image_with_landmarks = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        current_height = image_with_landmarks.shape[0]
        scale_factor = target_height / current_height
        image_resized = cv2.resize(image_with_landmarks, (0, 0), fx=scale_factor, fy=scale_factor)
        
        return image_resized

    left_image_path = "UPLOAD_FOLDER/left.png"
    right_image_path= "UPLOAD_FOLDER/left.png"

    embedding1 = get_face_embedding(left_image_path)
    embedding2 = get_face_embedding(right_image_path)

    image_paths = ["UPLOAD_FOLDER/left.png", "UPLOAD_FOLDER/right.png"]

    images = [process_image(image_path) for image_path in image_paths]
    concatenated_image = np.concatenate(images, axis=1)

    text_to_display = f"{str(final_match_percentage)}% Match"
    cv2.putText(concatenated_image, text_to_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 73), 2, cv2.LINE_AA)
    cv2.putText(concatenated_image, "Resc-U | AI for Good", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 73), 2, cv2.LINE_AA)

    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    output_path = os.path.join(static_dir, 'concatenated_image.jpg')
    try:
        cv2.imwrite(output_path, concatenated_image)
        print(f"Image saved successfully at {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
    return render_template('index.html', image_file='static/concatenated_image.jpg')

if __name__ == '__main__':
    app.run(debug=False)
