import streamlit as st
import joblib
import re
from transformers import pipeline
from scipy.sparse import hstack

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Twitter Disaster Tweet Classifier",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------
st.markdown("""
<style>

.main{
    background-color:#f4f7fc;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

h1{
    color:white;
}

.header{
    background:linear-gradient(90deg,#1e3c72,#2a5298,#6a11cb);
    padding:30px;
    border-radius:18px;
    text-align:center;
    color:white;
    box-shadow:0px 6px 18px rgba(0,0,0,0.25);
    margin-bottom:25px;
}

.card{
    background:white;
    border-radius:15px;
    padding:20px;
    box-shadow:0px 3px 12px rgba(0,0,0,0.15);
}

.metric-card{
    background:white;
    border-radius:15px;
    padding:20px;
    text-align:center;
    box-shadow:0px 3px 12px rgba(0,0,0,0.15);
}

.disaster{
    background:#ffebee;
    color:#b71c1c;
    padding:18px;
    border-radius:12px;
    font-size:22px;
    font-weight:bold;
    text-align:center;
}

.safe{
    background:#e8f5e9;
    color:#1b5e20;
    padding:18px;
    border-radius:12px;
    font-size:22px;
    font-weight:bold;
    text-align:center;
}

textarea{
    font-size:18px !important;
}

.stButton>button{
    width:100%;
    background:#1976d2;
    color:white;
    font-size:20px;
    border-radius:12px;
    height:55px;
    border:none;
}

.stButton>button:hover{
    background:#0d47a1;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# SIDEBAR
# ---------------------------------
with st.sidebar:

    st.title("📊 Project Details")

    st.success("Disaster Tweet Classification")

    st.markdown("---")

    st.write("### 🤖 Algorithm")
    st.info("Logistic Regression")

    st.write("### 📝 Feature Extraction")
    st.info("TF-IDF")

    st.write("### ❤️ Sentiment")
    st.info("Hugging Face DistilBERT")

    st.write("### 💻 Framework")
    st.info("Streamlit")

    st.markdown("---")

    st.write("### 🎯 Project Workflow")

    st.write("""
    ✔ Text Cleaning

    ✔ TF-IDF

    ✔ Hugging Face Sentiment

    ✔ Logistic Regression

    ✔ Prediction
    """)

# ---------------------------------
# HEADER
# ---------------------------------
st.markdown("""
<div class='header'>

<h1>🚨 Twitter Disaster Tweet Classifier</h1>

<h4>
Machine Learning using TF-IDF + Logistic Regression + Hugging Face
</h4>

</div>
""", unsafe_allow_html=True)

# ---------------------------------
# LOAD MODEL
# ---------------------------------
try:

    model = joblib.load("../models/disaster_classifier.pkl")

    tfidf = joblib.load("../models/tfidf_vectorizer.pkl")

    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

except Exception as e:

    st.error(e)

    st.stop()

# ---------------------------------
# CLEANING FUNCTION
# ---------------------------------
def clean_text(text):

    text=text.lower()

    text=re.sub(r"http\S+","",text)

    text=re.sub(r"www\S+","",text)

    text=re.sub(r"@\w+","",text)

    text=re.sub(r"#","",text)

    text=re.sub(r"[^a-zA-Z\s]","",text)

    text=re.sub(r"\s+"," ",text).strip()

    return text
# ---------------------------------
# MAIN LAYOUT
# ---------------------------------

left_col, right_col = st.columns([2, 1])

with left_col:

    st.markdown("## ✍️ Enter Tweet")

    tweet = st.text_area(
        "",
        height=180,
        placeholder="Example:\nMassive earthquake destroyed several buildings and rescue teams have been deployed..."
    )

    predict = st.button("🚀 Predict Tweet")

with right_col:

    st.markdown("## ℹ️ About")

    st.markdown("""
    This dashboard predicts whether a tweet is related to a disaster.

    **Workflow**

    • Clean Text

    • Hugging Face Sentiment

    • TF-IDF Vectorization

    • Logistic Regression

    • Final Prediction
    """)

# ---------------------------------
# PREDICTION
# ---------------------------------

if predict:

    if tweet.strip() == "":

        st.warning("⚠️ Please enter a tweet.")

    else:

        cleaned = clean_text(tweet)

        # -------------------------
        # Sentiment
        # -------------------------

        sentiment_result = sentiment_pipeline(cleaned)

        sentiment_label = sentiment_result[0]["label"]

        sentiment_score = sentiment_result[0]["score"]

        sentiment_encoded = 1 if sentiment_label == "POSITIVE" else 0

        # -------------------------
        # TF-IDF
        # -------------------------

        tfidf_features = tfidf.transform([cleaned])

        # -------------------------
        # Combine Features
        # -------------------------

        final_features = hstack(
            [tfidf_features, [[sentiment_encoded]]]
        )

        # -------------------------
        # Prediction
        # -------------------------

        prediction = model.predict(final_features)[0]

        probability = model.predict_proba(final_features)[0]

        confidence = max(probability) * 100

        st.markdown("---")

        st.markdown("## 🎯 Prediction Result")

        if prediction == 1:

            st.markdown(
                """
                <div class='disaster'>
                🚨 DISASTER TWEET
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class='safe'>
                ✅ NON-DISASTER TWEET
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                f"""
                <div class='metric-card'>
                <h3>🎯 Confidence</h3>
                <h2>{confidence:.2f}%</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            emoji = "😊" if sentiment_label == "POSITIVE" else "😔"

            st.markdown(
                f"""
                <div class='metric-card'>
                <h3>❤️ Sentiment</h3>
                <h2>{emoji} {sentiment_label}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                f"""
                <div class='metric-card'>
                <h3>📊 HF Score</h3>
                <h2>{sentiment_score*100:.2f}%</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.progress(confidence / 100)

        st.caption(f"Model Confidence : {confidence:.2f}%")
                # ---------------------------------
        # CLEANED TWEET
        # ---------------------------------

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("## 🧹 Cleaned Tweet")

        st.markdown(
            f"""
            <div class='card'>
            {cleaned}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------
        # PREDICTION EXPLANATION
        # ---------------------------------

        st.markdown("## 📖 Prediction Summary")

        if prediction == 1:

            st.info("""
**Model Prediction:** Disaster Tweet

The machine learning model predicts that this tweet is related to a disaster event.

This prediction is based on:

- TF-IDF text features
- Hugging Face sentiment feature
- Logistic Regression classifier
""")

        else:

            st.success("""
**Model Prediction:** Non-Disaster Tweet

The machine learning model predicts that this tweet is not related to a disaster.

This prediction is based on:

- TF-IDF text features
- Hugging Face sentiment feature
- Logistic Regression classifier
""")

# ---------------------------------
# FOOTER
# ---------------------------------

st.markdown("---")

st.markdown(
"""
<div class='footer'>

🚨 <b>Twitter Disaster Tweet Classification System</b><br>

Developed using <b>Python • Streamlit • Scikit-learn • Hugging Face</b>

</div>
""",
unsafe_allow_html=True
)