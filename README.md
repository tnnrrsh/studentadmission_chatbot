# studentadmission_chatbot

# MARbot – UiTM Chatbot for New Student Registration

MARbot is a rule-based and deep learning-enhanced chatbot built with Streamlit. It helps new UiTM students get quick answers to registration-related questions using hybrid TF-IDF and T5 model retrieval.

## Features
- Pattern-matching via TF-IDF
- Contextual understanding via T5 encoder embeddings
- Malay language preprocessing using Malaya
- Responsive chatbot interface with custom bubble styling
- Typo correction via probability model
- Chat history handling

## Deployment
The chatbot is deployed using Streamlit Community Cloud. Click the link to interact.

## How to Run Locally
1. Clone the repo
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Run the chatbot:
    ```
    streamlit run app.py
    ```

## File Structure
```
├── app.py                          # Main Streamlit app
├── chatbot_cleaned_with_embeddings.csv  # Cleaned dataset with T5 embeddings
├── tfidf_vectorizer.pkl           # Saved TF-IDF model
├── requirements.txt               # Dependencies list
├── README.md                      # Project overview
```

## Credits
Developed by: Tuan Nur Ariesha
