import os
import numpy as np
import time
from datetime import datetime
from flask import Flask,render_template,request,jsonify,session,redirect,url_for
from flask_login import LoginManager,UserMixin,login_user,logout_user,current_user,login_required
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = "static/uploads"
os.makedirs(app.config['UPLOAD_FOLDER'],exist_ok = True)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
CORS(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
MODEL_PATH = os.path.join("model","waste_classifier_model.keras")
IMG_SIZE = (224,224)
labels = ["metal","organic","paper","plastic"]
BIN_MAP = {
  "metal":{"bin":"Grey Bin","color":"#808080"},
  "organic": {"bin":"Green Bin","color":"#32CD32"},
  "paper": {"bin":"Blue Bin","color":"#1E90FF"},
  "plastic": {"bin":"Yellow Bin","color":"#FFD700"}

}
print("Model is Loading...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
class User(UserMixin,db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer,primary_key = True)
    username = db.Column(db.String(50),unique = True,nullable = False)
    email = db.Column(db.String(100),unique = True,nullable = False)
    hash_password = db.Column(db.String(255),nullable = False)
    created_at = db.Column(db.DateTime,default = datetime.utcnow)
    predictions = db.relationship('Prediction',backref = "user",lazy = True)
    def set_password(self,password):
        self.hash_password = bcrypt.generate_password_hash(password).decode('utf-8')
    def check_password(self,password):
        return bcrypt.check_password_hash(self.hash_password,password)
    def to_json(self):
        return {
            "id":self.id,
            "username":self.username,
            "email":self.email
        }
class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer,primary_key = True)
    user_id = db.Column(db.Integer,db.ForeignKey('users.id'),nullable = False)
    image_path = db.Column(db.String(255),nullable = False)
    predicted_label = db.Column(db.String(50),nullable = False)
    confidence = db.Column(db.Float,nullable = False)
    bin_assign = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime,default = datetime.utcnow)
    def to_json(self):
        return {
            "id":self.id,
             "prediction":self.predicted_label,
             "confidence_interval":round(self.confidence*100,2),
             "bin_assigned":self.bin_assign,
             "timestamp":self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
def preprocess_image(img:Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img,dtype = np.float32)/255.0
    return np.expand_dims(arr,axis = 0)
def run(img:Image.Image):
    if model is None:
        return "Unknown",0.0
    img = preprocess_image(img)
    preds = model.predict(img)[0]
    idx = int(np.argmax(preds))
    label = labels[idx]
    confidence = float(preds[idx])
    return label,confidence
def save_image(img:Image.Image,label:str) -> str:
    filename = f"{label}_{int(time.time())}.jpg"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
    img.save(save_path)
    return save_path
@app.route("/")
def index():
    return redirect(url_for("login"))
@app.route("/api/auth/register",methods = ["GET","POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    data = request.get_json()
    username = data.get("username","").strip()
    email = data.get("email","").strip()
    password = data.get("password","")
    if not username or not email or not password:
        return jsonify({"error": "All fields are required."}),400
    if User.query.filter_by(username = username).first():
        return jsonify({"error": "Username already exists."}),409
    if User.query.filter_by(email = email).first():
        return jsonify({"error":"email already registered"}),409
    user = User(username = username,email = email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message":"Registration successful!"}),201
@app.route("/api/auth/login",methods = ['GET','POST'])
def login():
    if request.method == "GET":
        return render_template("login.html")
    data = request.get_json()
    username = data.get("username","").strip()
    password = data.get("password","").strip()
    user = User.query.filter_by(username = username).first()
    if not user or not user.check_password(password):
        return jsonify({"error":"Invalid username or password."}),401
    login_user(user)
    return jsonify({"message":"Login Successful",
         "user":user.to_json()
    }),200
@app.route("/api/auth/logout",methods = ['GET','POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))
@app.route("/dashboard",methods = ["GET"])
@login_required
def dashboard():
    return render_template("dashboard.html")
@app.route("/api/auth/me",methods = ['GET'])
@login_required
def me():
    return jsonify({"user":current_user.to_json()}),200
@app.route("/api/predict",methods = ['POST'])
@login_required
def predict():
    if 'image' not in request.files:
        return jsonify({"error":'No image provided'}),400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}),400
    try:
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error":"Invalid image file."}),400
    label,confidence = run(img)
    bin_info = BIN_MAP.get(label,{"bin":"Unknown","color":"#ccc"})
    image_path = save_image(img,label)
    pred = Prediction(
        user_id = current_user.id,
        image_path = image_path,
        predicted_label = label,
        confidence = confidence,
        bin_assign = bin_info["bin"]
    )
    db.session.add(pred)
    db.session.commit()
    return jsonify({
        "label":label,
        "confidence":round(confidence*100,2),
        "bin":bin_info["bin"],
        "bin_color":bin_info["color"],
        "image_path":image_path,
        "timestamp":pred.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    }),200
@app.route("/api/predictions/history",methods = ['GET'])
@login_required
def prediction_history():
    preds = Prediction.query.filter_by(user_id = current_user.id).order_by(Prediction.timestamp.desc()).limit(50).all()
    return jsonify({"predictions":[p.to_json() for p in preds]}),200
@app.route("/api/predictions/stats",methods = ['GET'])
@login_required
def prediction_stats():
    preds = Prediction.query.filter_by(user_id = current_user.id).all()
    total = len(preds)
    category_counts = {}
    for p in preds:
        category_counts[p.predicted_label] = category_counts.get(p.predicted_label,0) + 1
    recyclable = total - category_counts.get("organic",0)
    recyclable_rate = round((recyclable/total)*100,2) if total > 0 else 0
    return jsonify({
        "total":total,
        "category_counts":category_counts,
        "recyclable_rate":recyclable_rate
    }),200
@app.route("/api/health",methods = ['GET'])
def health():
    return jsonify({
        "status":"running",
        "model_loaded": model is not None,

    }),200
@app.route('/api/predictions/linechart',methods = ['GET'])
@login_required
def linechart_data():
    from sqlalchemy import func,cast,Date
    results = (
        db.session.query(
            cast(Prediction.timestamp,Date).label("date"),
            func.count(Prediction.id).label("count")
        )
        .filter_by(user_id = current_user.id).group_by(cast(Prediction.timestamp,Date)).order_by(cast(Prediction.timestamp,Date)).limit(7).all()
    )
    return jsonify({
        "labels":[str(r.date) for r in results],
        "values":[r.count for r in results]
    }),200
@app.route('/api/predictions/barchart',methods = ['GET'])
@login_required
def barchart_data():
    preds = Prediction.query.filter_by(user_id = current_user.id).all()
    counts = {}
    for p in preds:
        counts[p.predicted_label] = counts.get(p.predicted_label,0) + 1
        return jsonify({
            "labels":list(counts.keys()),
            "values":list(counts.values())
        }),200

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username = "admin").first():
            admin = User(username = "admin",email = "admin@waste.local")
            admin.set_password("admin123")
            db.session.add(admin)
            db.session.commit()
    app.run(port = 5000,debug = True)