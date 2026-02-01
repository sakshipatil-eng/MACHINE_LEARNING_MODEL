# -------------------------------
# Streamlit Spam Email Detector (Final Version)
# -------------------------------

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# Dataset: Spam + Non-Spam Emails
# -------------------------------
em = [
    # Spam Emails
    "Win money now! Claim your free prize today!",
    "Urgent: Your bank account has been locked. Verify immediately.",
    "Limited time offer! Get huge discounts today.",
    "Congratulations! You have won a cash reward.",
    "Free entry to win a brand new iPhone.",
    "Earn money from home without investment.",
    "Your account has been compromised. Login immediately.",
    "Exclusive offer just for you. Buy now!",
    "You are selected for a free vacation.",
    "Claim your reward before the deadline.",
    "Get rich quickly with this one simple trick.",
    "You won a lottery! Click to claim your prize.",
    "Urgent: Verify your identity to avoid account suspension.",
    "Special promotion: Buy 1 Get 1 free today.",
    "Your credit card has been approved. Apply now.",
    "Win a free trip to Dubai! Limited seats.",
    "Lowest interest rates on loans available now.",
    "You are eligible for a free gift card.",
    "Act now! Limited time investment opportunity.",
    "Congratulations! You won free cash rewards.",

    # Non-Spam Emails
    "Hey, are we meeting today?",
    "Let's have lunch tomorrow.",
    "Can you send me the notes from class?",
    "See you in the meeting at 10 am.",
    "How are you doing today?",
    "Please review the project report by tomorrow.",
    "Are you coming for the volleyball practice?",
    "Don't forget to submit your assignment.",
    "Thank you for your help on the project.",
    "Happy birthday! Wishing you a great day.",
    "Please find attached the documents you requested.",
    "Let me know your availability for the call.",
    "I enjoyed our discussion in class today.",
    "Can we reschedule our meeting to next week?",
    "Looking forward to our team outing this weekend.",
    "Please confirm your attendance for the workshop.",
    "Thank you for your support during the event.",
    "Can you share the lecture slides with me?",
    "I will be late to the meeting today.",
    "Congratulations on completing your project!"
]

lb = [1]*20 + [0]*20  # 1 = Spam, 0 = Not Spam

# -------------------------------
# Train TF-IDF + Naive Bayes on Entire Dataset
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(em)

model = MultinomialNB()
model.fit(X, lb)  # Train on all emails

# -------------------------------
# Streamlit GUI
# -------------------------------
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")
st.title("📧 Spam Email Detector")
st.write("Enter any email or message below to check if it's SPAM or NOT SPAM.")

# User input
uinp = st.text_area("Enter or paste your email/message here:")

# Predict button
if st.button("Check Spam"):
    if uinp.strip() == "":
        st.warning("⚠️ Please enter an email/message to check!")
    else:
        input_vector = vectorizer.transform([uinp])
        prediction = model.predict(input_vector)[0]
        prediction_proba = model.predict_proba(input_vector)[0]

        # Display result
        if prediction == 1:
            st.error(f"🚨 This email is SPAM!")
            st.info(f"Spam Probability: {prediction_proba[1]*100:.2f}%")
        else:
            st.success(f"✅ This email is NOT SPAM")
            st.info(f"Not Spam Probability: {prediction_proba[0]*100:.2f}%")
