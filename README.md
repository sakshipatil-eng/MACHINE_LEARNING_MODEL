# MACHINE_LEARNING_MODEL


# 📧 Spam Email Detector using Machine Learning

This project is a **web-based Spam Email Detector** built using **Python, Machine Learning, and Streamlit**.

It classifies an email or message as **SPAM** or **NOT SPAM** using **TF-IDF vectorization** and the **Naive Bayes algorithm**.



## 🎯 Project Objective

To build a simple and interactive application that:

* Analyzes email text
* Detects spam messages
* Displays prediction confidence
* Demonstrates basic text classification using ML


## ✨ Features

* Web interface using **Streamlit**
* Spam detection using **Naive Bayes classifier**
* Text vectorization with **TF-IDF**
* Displays probability score for prediction
* Instant classification with a button click



## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **Scikit-learn**
* **TF-IDF Vectorizer**
* **Naive Bayes (MultinomialNB)**



## 📂 Project Structure


spamemaildetector.py

README.md




## ⚙️ Requirements

Ensure Python 3.x is installed.

Install required libraries:

bash
pip install streamlit scikit-learn




## ▶️ How to Run the Application

1. Open terminal / command prompt
2. Navigate to the project directory
3. Run the app:

 bash

 streamlit run spam_email_detector.py

4. The app will open automatically in your browser.



## 🧠 How It Works

1. A dataset of **spam and non-spam emails** is used
2. Text data is converted into numerical features using **TF-IDF**
3. A **Naive Bayes model** is trained on the dataset
4. User input is classified as **SPAM** or **NOT SPAM**
5. Prediction probability is displayed



## 📊 Dataset Details

* Total emails: **40**

  * Spam: **20**
  * Non-Spam: **20**
  * Labels:

  * 1 → Spam
  * 0 → Not Spam



## 🚀 Future Enhancements

* Use a larger real-world dataset
* Add email file upload support
* Improve model accuracy
* Store prediction history
* Deploy on cloud (Streamlit Cloud / Heroku)



## 📜 License

This project is open-source and intended for **learning and academic purposes**.


## Output

<img width="1908" height="980" alt="Image" src="https://github.com/user-attachments/assets/34696323-1153-4eca-9b05-b8852d2c03c1" />





