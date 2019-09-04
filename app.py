import os
import threading
import time
from flask import Flask, request, render_template, send_from_directory
from FaceClassify import FaceClassification

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

ROOT = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = ['jpg', 'png', 'jpeg']

@app.route("/")
@app.route("/home")
def index():
    home_files, face_files, text, confs = FaceClassification.generate_home()
    return render_template('index.html', title = 'Home', examples = home_files, face_files = face_files, text=text, confs = confs)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def InternalServerError(e):
    home_files, face_files, text, confs = FaceClassification.generate_home()
    return render_template('index.html', title = 'Home', examples = home_files, face_files = face_files, text=text), 500

def allowed_file(file):
    return ('.') in file and file.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files['file'] 
   
    if file and allowed_file(file.filename):
        image_target = os.path.join(ROOT, 'images/')
        face_target = os.path.join(ROOT, 'faces/')
        model_path = os.path.join(ROOT, 'model')

        if not os.path.isdir(image_target):
                os.mkdir(image_target)
        if not os.path.isdir(face_target):
                os.mkdir(face_target)
   
        image_name, face_names, num_faces = FaceClassification.process_image(file, image_target, face_target)

        predictions = FaceClassification.classify_image(model_path, face_target, face_names)
        return render_template("display.html", image_name = image_name, predictions = predictions,
                               face_names = face_names, num_faces = num_faces, title = 'Prediction')
    else:
        return render_template("error.html", title = 'Home', example='blank.jpg')

def delayed_delete(delay, path):
    time.sleep(delay)
    os.remove(path)
    return

@app.route('/<filename>')   
def load_home(filename):
    return send_from_directory('static/home_images', filename)

@app.route('/upload/<filename>')
def send_image(filename):
    del_thread = threading.Thread(target=delayed_delete, args=(5, os.path.join(ROOT, 'images/') + filename))
    del_thread.start()
    return send_from_directory('images', filename)

@app.route('/predictions/<facename>')
def send_face(facename):
#    del_thread = threading.Thread(target=delayed_delete, args=(5, os.path.join(ROOT, 'faces/') + facename))
#    del_thread.start()
    return send_from_directory('faces', facename)

if __name__ == "__main__":
    app.run('localhost', port=5555)