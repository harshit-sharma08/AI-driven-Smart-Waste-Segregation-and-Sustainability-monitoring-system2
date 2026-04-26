# ♻️ AI-Driven Smart Waste Segregation & Sustainability Monitoring System

> An intelligent IoT-integrated system that uses a MobileNet deep learning model to classify waste in real time and monitor sustainability metrics through a web dashboard.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
- [AI Model](#ai-model)
- [Database Schema](#database-schema)
- [Hardware Setup](#hardware-setup)
- [Contributors](#contributors)

---

## 🌐 Overview
The AI-Driven Smart Waste Segregation and Sustainability Monitoring System is an innovative university-level hardware project that integrates computer vision, IoT devices, and web technologies to automate waste classification at the source.

It utilizes a MobileNetV2-based convolutional neural network (CNN) to analyze camera input and categorize waste into four types: Plastic, Paper, Organic, and Metal. Each prediction is recorded in a database, and real-time sustainability insights are displayed through an interactive web dashboard.

The system also includes user authentication features and is designed to control physical waste sorting mechanisms using an ESP32 microcontroller, enabling automated and efficient waste management.

## ▶️ How to Run (Detailed)

1. Clone the repository and navigate into the project folder.
2. Install all required Python dependencies using `pip install -r requirements.txt`.
3. Ensure the trained MobileNet model file is placed correctly inside the `model/` directory.
4. Configure environment variables in the `.env` file.
5. Start the Flask development server using `python app.py`.
6. Open your browser and visit `http://127.0.0.1:5000`.
7. Register a new account or use the default admin credentials to log in.

> ⚠️ Make sure your system has Python 3.8+ and pip installed.

---

## ✨ Features

| Feature | Status | Description |
|---|---|---|
| 🤖 AI Waste Classification | ✅ Done | MobileNetV2 classifies waste into 4 categories |
| 📸 Image Upload & Prediction | ✅ Done | Upload image from browser, get instant result |
| 📊 Sustainability Dashboard | ✅ Done | Charts for waste trends and recycling rate |
| 🔐 User Authentication | ✅ Done | Register, login, logout with encrypted passwords |
| 🗃️ Prediction Logging | ✅ Done | Every prediction stored in database with timestamp |
| 📷 ESP32 Camera Integration | 🔄 Planned | Live camera feed from ESP32 for real-time scanning |
| ⚙️ Hardware Bin Mechanism | 🔄 Planned | ESP32 controls servo motors for physical segregation |
| 📡 Bin Fill Level Monitoring | 🔄 Planned | Ultrasonic sensor monitors how full each bin is |

---

## 🏗️ System Architecture

```
[ Image Upload / ESP32 Camera ]
              │
              ▼
   [ Flask Backend (app.py) ]
         │           │
         ▼           ▼
  [ MobileNet     [ SQLite DB ]
    .keras Model]  (users + predictions)
         │
         ▼
  [ Prediction Result ]
         │
         ├──► [ Web Dashboard ] ── (Charts, History, Stats)
         │
         └──► [ ESP32 ] ── (Servo Motors → Physical Bins)
                    │
                    └──► [ Ultrasonic Sensor → Bin Fill Level ]
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| AI Model | TensorFlow / Keras, MobileNetV2, NumPy, Pillow |
| Backend | Flask, Flask-Login, Flask-SQLAlchemy, Flask-Bcrypt |
| Frontend | HTML5, CSS3, JavaScript, Chart.js |
| Database | SQLite |
| Hardware | ESP32, Servo Motors, Ultrasonic Sensor, Camera Module |

---

## 📁 Folder Structure

```
waste-segregation/
│
├── app.py                          # Main Flask backend
├── requirements.txt                # Python dependencies
├── .env                            # Secret keys (not on GitHub)
├── .gitignore                      # Files excluded from GitHub
│
├── model/
│   └── mobilenet_waste_classifier.keras   # Trained MobileNet model (not on GitHub)
│
├── templates/
│   ├── login.html                  # Login page
│   ├── register.html               # Register page
│   └── dashboard.html              # Main dashboard
│
└── static/
    └── uploads/                    # Saved prediction images
```

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/AI-Smart-Waste-Segregation.git
cd AI-Smart-Waste-Segregation
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Create `.env` file**
```
SECRET_KEY=your_random_secret_key_here
```

**4. Place your model file**

The model file is too large for GitHub so place it manually inside the `model/` folder:
```
model/mobilenet_waste_classifier.keras
```

**5. Run the server**
```bash
python app.py
```

**6. Open in browser**
```
http://127.0.0.1:5000
```

A default admin account is created automatically on first run:
- Username: `admin`
- Password: `admin123`

---

## 🤖 AI Model

| Detail | Value |
|---|---|
| Base Model | MobileNetV2 (ImageNet weights) |
| Input Size | 224 × 224 × 3 |
| Output Classes | 4 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Format | `.keras` |

### Waste Categories

| Label | Bin |
|---|---|
| 🟡 Plastic | Yellow Bin |
| 🔵 Paper | Blue Bin |
| 🟢 Organic | Green Bin |
| ⚫ Metal | Grey Bin |

---

## 🗃️ Database Schema

### `users` Table
| Column | Type | Description |
|---|---|---|
| id | INTEGER | Auto increment primary key |
| username | VARCHAR | Unique username |
| email | VARCHAR | Unique email |
| password_hash | VARCHAR | Bcrypt hashed password |
| created_at | DATETIME | Registration timestamp |

### `predictions` Table
| Column | Type | Description |
|---|---|---|
| id | INTEGER | Auto increment primary key |
| user_id | INTEGER | References users.id |
| image_path | VARCHAR | Path to saved image |
| predicted_label | VARCHAR | e.g. Plastic |
| confidence | FLOAT | e.g. 0.94 means 94% |
| bin_assigned | VARCHAR | e.g. Yellow Bin |
| timestamp | DATETIME | When prediction was made |

---

## ⚙️ Hardware Setup

### Components Required

| Component | Quantity | Purpose |
|---|---|---|
| ESP32 | 1 | Main microcontroller + WiFi |
| Servo Motor (SG90) | 4 | One per bin for physical segregation |
| Ultrasonic Sensor (HC-SR04) | 4 | Monitor fill level of each bin |
| IR Sensor | 1 | Detect when waste is placed |
| ESP32-CAM | 1 | Capture waste image |
| 5V Power Supply | 1 | Power the entire system |

### How It Works

1. Person places waste item on the platform
2. IR Sensor detects the item and signals ESP32
3. ESP32-CAM captures image and sends it to Flask backend
4. Flask runs MobileNet and returns the predicted label
5. ESP32 receives the result and activates the correct servo motor
6. Servo opens the correct bin flap for that waste category
7. Ultrasonic sensors continuously monitor bin fill levels
8. Fill level data is sent to the dashboard for monitoring
9.  Flask runs MobileNet and returns the predicted label


## 🚀 Future Improvements

- Real-time ESP32 camera streaming integration
- Cloud deployment (AWS / Azure)
- Mobile app support

---

## 📄 License

This project is developed for educational purposes and is free to use for learning and research.

## 👥 Contributors


| Akshay Kumar |

| Harshit Sharma |

| Rachit Saxena |


---


> ⭐ If you find this project helpful, please give it a star!
