import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords

st.title('Sentiment Analyser App')
st.write('This app uses the Hugging Face Transformers [sentiment analyser](https://huggingface.co/course/chapter1/3?fw=tf) library to clasify the sentiment of your input as postive or negative. The web app is built using [Streamlit](https://docs.streamlit.io/en/stable/getting_started.html).')
st.write(
    'To find out how this app was developed, please check out my [Medium post](https://medium.com/@rtkilian/deploy-and-share-your-sentiment-analysis-app-using-streamlit-sharing-2ba3ca6a3ead). To see my source code, have a look at my [GitHub repo](https://github.com/rtkilian/streamlit-huggingface).')

st.write('*Note: it will take up to 30 seconds to run the app.*')

form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your text')
submit = form.form_submit_button('Submit')

nltk.download('stopwords')
set(stopwords.words('english'))

if submit:
    text_final = ''.join(c for c in user_input if not c.isdigit())
    
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound'])/2, 2)
    if compound<0.00:
        score = compound*-1
        score = score*100
        st.success(f'Negative sentiment (score: {compound} Negative)')
    if compound>=0.00:
        score = score*100
        st.success(f'Positive sentiment (score: {compound} Positive)')