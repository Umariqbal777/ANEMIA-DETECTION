# 🩸 Anemia Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Flask](https://img.shields.io/badge/Flask-web--framework-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

A machine learning-based system for predicting the likelihood of anemia using patient health parameters.  
The project evaluates multiple ML algorithms — **Logistic Regression**, **Decision Tree**, **Random Forest**, **SVM**, **Naive Bayes**, and **Gradient Boosting** — and deploys the **Gradient Boosting Classifier** for its superior accuracy.

---

## 📌 Features
- Multi-model training and performance comparison
- Gradient Boosting Classifier as the final deployed model
- Flask-based web application for easy interaction
- Model saved as `model.pkl` for real-time predictions
- User-friendly input form for patient data

---

## 🛠 Tech Stack
- **Programming Language:** Python  
- **Libraries:** scikit-learn, pandas, numpy, matplotlib  
- **Framework:** Flask  
- **Version Control:** Git & GitHub  

---

## 📊 Dataset
The model is trained on a dataset containing patient health records with attributes relevant to anemia detection (e.g., hemoglobin levels, red blood cell count, gender, etc.).

*Note:* You can replace the dataset with your own, as long as the features match the training configuration.

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/anemia-detection-ml.git
cd anemia-detection-ml
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask App
```bash
python app.py
```

The app will start locally at:  
```
http://127.0.0.1:5000/
```

---

## 📂 Project Structure
```
.
├── app.py                # Flask application
├── model.pkl             # Saved ML model
├── requirements.txt      # Dependencies
├── train.py/             # Code to train the machine 
├── templates/            # HTML templates
└── README.md             # Project documentation
```

---

## 📈 Model Performance
The **Gradient Boosting Classifier** achieved the highest accuracy among tested models.  
| Model                | Accuracy |
|----------------------|----------|
| Loinear Regression   | 0.9919354838709677 |
| Decision Tree        | 1.0                |
| Random Forest        | 1.0                |
| SVM                  | 0.9798387096774194 |
| Naive Bayes          | 0.9395161290322581 |
| Gradient Boosting    | 1.0                |



---

## 💡 Use Cases
- Early screening for anemia
- Healthcare decision support
- Academic demonstration of ML in medical diagnostics

---

## 📜 License
This project is licensed under the **MIT License** — feel free to modify and use it.

---

## 🤝 Contributing
Contributions are welcome!  
1. Fork the repository  
2. Create a new branch (`feature-branch`)  
3. Commit your changes  
4. Push to your fork  
5. Open a Pull Request  

---

## 📬 Contact
Umar Iqbal
📧 iqbalumar131@gmail.com
