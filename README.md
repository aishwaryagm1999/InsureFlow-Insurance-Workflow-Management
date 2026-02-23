
# 🏦 Insurance Workflow Management System

### Full-Stack Intelligent Insurance Platform with Fraud Detection & NLP Integration

---

## 📌 Project Overview

The **Insurance Workflow Management System** is a full-stack web application developed using **Python Flask**, integrating traditional insurance management workflows with **Machine Learning (Fraud Detection)** and **Natural Language Processing (Customer Sentiment Analysis)**.

The platform simulates a real-world insurance ecosystem including:

* User authentication and profile management
* Multi-category policy registration
* QR-based policy verification
* Claim processing and renewal system
* SMS notifications via Twilio API
* Fraud detection using Decision Tree classifier
* Customer satisfaction classification using NLP

This project demonstrates the integration of **Web Development + Database Systems + Machine Learning + API Integration** into a unified intelligent system.

---

# 🎯 Problem Statement

Traditional insurance management systems:

* Require manual claim verification
* Lack automated fraud detection
* Do not integrate sentiment analytics
* Do not provide dynamic renewal alerts
* Lack transparent policy verification mechanisms

This project addresses these limitations by building an **intelligent, automated insurance workflow platform** capable of:

* Detecting fraudulent claims using ML
* Classifying customer feedback using NLP
* Generating QR codes for policy validation
* Sending real-time SMS notifications
* Managing the complete lifecycle of insurance policies

---

# 🏗️ System Architecture

```
Frontend (HTML/CSS + Jinja)
        ↓
Flask Application Layer
        ↓
Business Logic Layer
        ↓
Database (SQLite + JSON)
        ↓
ML Modules (Fraud + NLP)
        ↓
External API (Twilio SMS)
```

---

# 💻 Technology Stack

## 🔹 Backend

* Python 3.x
* Flask (Web Framework)
* SQLAlchemy (ORM)
* SQLite (Relational DB)
* JSON-based structured storage

## 🔹 Frontend

* HTML5
* CSS3
* Jinja2 Templating Engine
* Responsive UI Design

## 🔹 Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Joblib (Model Serialization)

## 🔹 External APIs

* Twilio SMS API
* QR Code Generation Library

---

# 🧠 Artificial Intelligence & Machine Learning Components

---

## 1️⃣ Fraud Detection System

### 🎯 Objective

Predict whether a submitted insurance claim is fraudulent based on structured features.

### 📊 Dataset

* `insurance_claims.csv`
* Contains structured attributes:

  * Demographic features
  * Policy details
  * Incident type
  * Claim amounts
  * Vehicle details
  * Injury & property damage info

### 🧮 Algorithm Used

* **Decision Tree Classifier**

### 📈 Model Performance

* Accuracy: **77%**
* Trained and serialized using `joblib`
* Tested using structured policy samples

### ⚙️ Libraries Used

* `sklearn.tree.DecisionTreeClassifier`
* `sklearn.model_selection.train_test_split`
* `sklearn.metrics`
* `pandas`
* `numpy`
* `joblib`

### 🧪 Workflow

1. Load structured claim dataset
2. Encode categorical features
3. Split into train/test sets
4. Train Decision Tree model
5. Evaluate accuracy
6. Serialize model (.joblib)
7. Load model for prediction

---

## 2️⃣ Customer Satisfaction Detection (NLP)

### 🎯 Objective

Classify customer feedback as:

* Positive
* Negative

### 🧠 NLP Techniques Used

* Text preprocessing
* Basic polarity classification
* Feedback storage in `feedback.txt`

### 📚 Libraries Used

* `sklearn.feature_extraction.text`
* `sklearn.linear_model` / classification model
* `pandas`
* `re` (regular expressions)

### ⚙️ Workflow

1. User submits feedback
2. Feedback stored in text file
3. Run sentiment classifier
4. Label feedback polarity

---

# 🔐 User Authentication & Database Design

## 🔹 User Management

* Signup
* Login
* Logout
* Session management
* Profile photo upload
* SQLite database storage

### Database Tools Used

* `SQLAlchemy`
* `Flask-Login`
* SQLite

User details stored in:

```
instances/users.db
```

---

# 📋 Policy Management System

## Supported Insurance Types

* Car Insurance
* Bike Insurance
* Travel Insurance
* Health Insurance
* Business Insurance
* Home Insurance

---

## 🔄 Policy Lifecycle

### Policy Creation

* Submit structured form
* Generate unique policy ID
* Generate QR code
* Save policy in `policies.json`
* Send SMS confirmation

### QR Code Verification

Each policy contains a unique QR code linking to structured policy data.

Libraries used:

* `qrcode`
* `Pillow`

---

### Claim Processing

* User submits claim amount
* Insurance amount reduced
* Claim recorded in policy
* SMS alert triggered

---

### Renewal System

* Update expiry date
* SMS renewal notification
* Expiry tracking mechanism

---

# 📲 SMS Notification System

Integrated using **Twilio API**.

Notifications triggered for:

* Policy creation
* Claim submission
* Expiry alerts
* Renewal confirmation

Libraries used:

* `twilio.rest.Client`

---

# 🗄️ Data Storage Architecture

| Component     | Storage Type            |
| ------------- | ----------------------- |
| User Accounts | SQLite (Relational DB)  |
| Policies      | JSON File               |
| Feedback      | Text File               |
| Fraud Model   | Joblib Serialized Model |
| QR Codes      | Static Folder Images    |

---

# 🛠️ Python Libraries Used (Detailed)

### Web & Backend

* flask
* flask_sqlalchemy
* flask_session
* werkzeug

### Database

* sqlite3
* SQLAlchemy ORM

### Machine Learning

* scikit-learn
* pandas
* numpy
* joblib

### NLP

* sklearn.feature_extraction.text
* sklearn.naive_bayes / logistic regression (if used)

### Utility

* qrcode
* pillow
* twilio
* datetime
* uuid
* json
* os
* re

---

# ▶️ How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/insurance-workflow-management.git
cd insurance-workflow-management
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run Flask Application

```bash
python app.py
```

Open browser:

```
http://localhost:5000
```

---

### 4️⃣ Run Fraud Detection Model

```bash
jupyter notebook fraud_detection_model.ipynb
```

---

### 5️⃣ Run Sentiment Classifier

```bash
python customer_satisfaction.py
```

---

# 📊 Experimental Results

| Model                         | Accuracy              |
| ----------------------------- | --------------------- |
| Decision Tree Fraud Detection | 77%                   |
| Sentiment Classifier          | Binary Classification |

---

# 🔬 Advanced Concepts Demonstrated

* Full-stack application architecture
* REST-like route management
* ORM-based relational database design
* Structured + unstructured data handling
* Model serialization & loading
* Fraud analytics integration
* Real-time notification systems
* QR-based verification system
* State-based workflow automation


## Overview
The Insurance Workflow Management System (IWMS) is a comprehensive web application designed to streamline the management of insurance policies, claims, and customer interactions while improving the efficiency and security of insurance workflows. The application provides seamless features such as user account creation, policy management, real-time alerts, customer feedback analysis, and fraud detection. Built using Python, HTML, and CSS, IWMS integrates machine learning (ML) and natural language processing (NLP) to offer a reliable, intelligent solution for insurance companies to enhance customer experience and operational security.

## Features

### 1. User Account Management
   - **User Registration and Login**: Users can register and log in, creating personalized accounts with secure credentials.
   - **Profile Management**: Users can view and manage their account information, with a navigation bar for easy access across the application.
   - **QR Code Verification**: Each policy is verified through a QR code, ensuring data security and authenticity.

### 2. Policy Management
   - **Policy Registration**: Users can explore various insurance policies (e.g., Car, Bike, Travel, Health, Business) and select options based on their needs.
   - **Detailed Forms**: Each policy type has a specific form to collect relevant information (e.g., car model, health conditions).
   - **Policy Overview**: Registered policies are displayed with key details, and users can initiate claims directly from their account dashboard.
   - **Real-time Notifications**: SMS alerts are sent to users when new policies are created, claims are submitted, and renewals or expirations are upcoming.

### 3. Coverage Details
   - **Comprehensive Information**: Users have access to detailed descriptions of each insurance policy type, including coverage details, inclusions, and exclusions.
   - **Centralized Info Centre**: All coverage information is accessible through an Info Centre, with consistent visual styling to ensure clarity.

### 4. Customer Feedback and Satisfaction Detection
   - **Feedback Collection**: Customers can submit feedback through a dedicated support interface.
   - **NLP-Based Sentiment Analysis**: Feedback is analyzed with TextBlob, an NLP library, to detect sentiment (positive, neutral, negative).
   - **Feedback Management**: Customer satisfaction is categorized based on feedback sentiment, allowing for efficient handling of support cases and follow-up.

### 5. Alert System
   - **Policy Creation Alerts**: SMS notifications are sent to users upon successful policy creation.
   - **Claim Status Notifications**: When a claim is processed, users receive updates via SMS.
   - **Expiration Reminders**: Automatic alerts are sent to users when their policies are nearing expiration, enabling timely renewals.
   - **Twilio Integration**: Twilio API is integrated for reliable SMS delivery, ensuring secure and instant communication with users.

### 6. Claim Processing and Fraud Detection
   - **Claim Collection**: Users can submit claims for their policies, which are stored and managed in the application’s database.
   - **Random Forest Classifier for Fraud Detection**: A machine learning model trained on insurance data identifies potential fraudulent claims by analyzing claim patterns and flagging suspicious entries.
   - **Policy Data Management**: All claim and policy updates are saved in JSON format, enhancing data accessibility and integrity.
   - **QR Code Generation**: QR codes are generated for each policy, offering an added layer of security for policy validation.

## Technical Specifications

### Libraries and Frameworks
- **Python**: Primary programming language for back-end logic, ML model training, and NLP processing.
- **HTML/CSS**: Front-end languages for structuring and styling the user interface.
- **OpenCV**: Used for image processing and handling any image-based requirements.
- **TensorFlow**: Facilitates the development of ML models, particularly the fraud detection model.
- **TextBlob**: NLP library used for sentiment analysis in customer feedback.
- **Twilio API**: Integrates SMS functionality to send real-time alerts to users.
- **LabelImg**: Image annotation tool used for creating labeled datasets for any visual data processing.

### Machine Learning
- **Random Forest Classifier**: Trained model used to detect fraudulent claims based on historical data. The classifier flags suspicious entries, which can be reviewed by administrators for further action.

### Data Storage and Processing
- **JSON**: Used to store policy and claim information in a structured format, ensuring efficient data retrieval and updates.
- **Database Interaction**: The system retrieves, stores, and updates user-specific policy information to maintain accurate records and provide up-to-date notifications.

## Usage

### User Journey
1. **Account Creation**: Users start by creating an account, with options to register or log in if they are returning users.
2. **Policy Selection**: After login, users can browse policy options and fill out forms to register their chosen policy.
3. **Claim Submission**: Users can submit claims on their policies and receive real-time notifications regarding their claim status.
4. **Feedback Submission**: Users can provide feedback, which is automatically analyzed for sentiment to help improve customer experience.
5. **Alerts**: SMS notifications inform users about policy updates, upcoming renewals, and other important actions.
6. **Policy Verification**: Users can verify policies using the generated QR code, ensuring security and authenticity.

### Developer and Management Interaction
- **User Information Management**: Developers can access user information and feedback to manage operations and enhance service quality.
- **Feedback Analysis**: Negative feedback triggers alerts to management, allowing for quick resolution of user issues.
- **Fraud Detection**: The fraud detection model identifies potentially fraudulent claims, enabling administrators to take necessary action.

