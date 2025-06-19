import os
import re
import torch
import malaya
import joblib
import pandas as pd
import streamlit as st
from PIL import Image
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PAGE CONFIG (must be first)
st.set_page_config(page_title="MARbot - UiTM Chatbot", layout="centered")

# Load Data
df = pd.read_csv("chatbot_cleaned_with_embeddings.csv")
df["embedding"] = df["embedding_list"].apply(eval).apply(torch.tensor)
embedding_matrix = torch.stack(df["embedding"].tolist())

# Preprocessing
stopwords = malaya.text.function.get_stopwords()
correction = malaya.spelling_correction.probability.Probability(corpus=stopwords)

def clean_text(text):
    text = str(text).lower()
    try:
        text = correction.correct(text)
    except:
        pass
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in stopwords]
    return " ".join(tokens)

# Load TF-IDF
if os.path.exists("tfidf_vectorizer.pkl"):
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
else:
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df["clean_question"].fillna(""))
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
tfidf_matrix = vectorizer.transform(df["clean_question"].fillna(""))

# Load T5 Encoder
try:
    tokenizer = T5Tokenizer.from_pretrained("malay-huggingface/t5-small-bahasa-cased")
    model = T5EncoderModel.from_pretrained("malay-huggingface/t5-small-bahasa-cased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            return model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu()

except Exception as e:
    st.warning("Model fallback active due to error loading T5. Using TF-IDF only.")
    def get_embedding(text):
        return torch.zeros(768)  # dummy embedding

# Chatbot Matching
def chatbot_match(query, threshold=0.1, alpha=0.7, beta=0.3, top_k=1):
    query_clean = clean_text(query)
    tfidf_vec = vectorizer.transform([query_clean])
    tfidf_scores = cosine_similarity(tfidf_vec, tfidf_matrix)[0]
    embed = get_embedding(query_clean)
    deep_scores = cosine_similarity(embed.unsqueeze(0), embedding_matrix)[0]
    combined_scores = (alpha * deep_scores) + (beta * tfidf_scores)
    best_indices = combined_scores.argsort()[-top_k:][::-1]
    if combined_scores[best_indices[0]] < threshold:
        return "Maaf, saya tidak pasti jawapannya. Sila cuba tanya dengan ayat lain."
    return df.iloc[best_indices[0]]["answer"]

# STYLING
st.markdown("""
<style>
.chat-container { display: flex; flex-direction: column; gap: 10px; padding: 10px; margin-top: 20px; }
.user-bubble {
    background-color: #c79ff2; color: white;
    align-self: flex-end;
    padding: 10px 15px;
    border-radius: 15px 0px 15px 15px;
    max-width: 75%;
    margin-left: auto;
}
.chat-container > div {margin-bottom: 10px; /* Add vertical space between bubbles */
}

.bot-bubble {
    background-color: white; color: black;
    align-self: flex-start;
    padding: 10px 15px;
    border: 1px solid #ccc;
    border-radius: 0px 15px 15px 15px;
    max-width: 75%;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

# HEADER & ICON
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo-chatbot.png", width=100)  # Make sure file exists
with col2:
    st.markdown("<h1 style='margin-bottom:0;'>MARbot - UiTM Chatbot</h1>", unsafe_allow_html=True)
    st.caption("Tanya apa-apa berkaitan pendaftaran pelajar baharu UiTM di sini!")

# CONVERSATION HISTORY
if "history" not in st.session_state:
    st.session_state.history = []

# DISPLAY BUBBLES
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, message in st.session_state.history:
    bubble_class = "user-bubble" if role == "Anda" else "bot-bubble"
    st.markdown(f'<div class="{bubble_class}">{message}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# INPUT FIELD
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Tanya soalan anda:")
    submitted = st.form_submit_button("Hantar")

if submitted and user_input:
    reply = chatbot_match(user_input)
    st.session_state.history.append(("Anda", user_input))
    st.session_state.history.append(("MARbot", reply))
    st.rerun()
