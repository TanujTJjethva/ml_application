from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib
import re
import requests
from bs4 import BeautifulSoup
from flask import jsonify

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_input(text, tokenizer, max_len):
    text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

def fetch_news(stock):
    url = f"https://www.moneycontrol.com/news/tags/{stock.lower()}.html"
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find all news list items
    articles = soup.find_all("li", class_="clearfix")
    
    for article in articles:
        title_tag = article.find("h2")
        desc_tag = article.find("p")
        
        # Skip invalid/nav items
        if title_tag and desc_tag:
            return desc_tag.text.strip()

    return None

def sentiment_prediction(stock):                 # Main function
    new_text = fetch_news(stock=stock)

    if new_text == None:
        return "No Content"

    max_len = 100

    model = load_model('models/sentiment_analysis_model.h5')
    le = joblib.load('models/sentiment_analysis_label_encoder.pkl')
    tokenizer = joblib.load('models/sentiment_analysis_tokenizer.pkl')

    input_data = prepare_input(new_text, tokenizer, max_len)

    prediction = model.predict(input_data)
    
    class_index = prediction.argmax(axis=1)[0]

    label = le.inverse_transform([class_index])[0]

    return jsonify({'news_sentiment': label})