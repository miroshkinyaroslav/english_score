import pandas as pd
import pickle
import pysrt as srt
import re
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk
from nltk.corpus import stopwords

from catboost import CatBoostClassifier, Pool

from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfTransformer,
                                             TfidfVectorizer)

st.set_page_config(layout="wide")

spacy.load("en_core_web_sm", disable=['parser', 'ner'])

nltk.download('stopwords')
nltk.download('punkt')

english_levels = ['A2/A2+', 'B1', 'B2', 'C1']
films_paths = ['streamlit/American Psycho 2000 1080p GER Blu-ray AVC DTS-HD MA 5.1-PEGASUS.srt',
               'streamlit/Breakfast at Tiffany\'s BDRip 1080p.eng.srt',
               'streamlit/The Lion King 1994 USA The Circle of Life Edition The Signature Collection 1080p Blu-ray AVC DTS-HD MA 7.1 - BluDragon_SDH.srt']

def show_main_info():
    st.title('Предсказание необходимого уровня английского языка для просмотра фильма')
    st.markdown(
    """
    ## Как работать с данным приложением?
    1. Загрузить файл субтитров. Формат srt. Посмотреть результат предсказания модели.
    Для удобства можно скачать некоторые субтитры ниже или с сайта https://www.opensubtitles.org/en/search/sublanguageid-eng
    2. Органический поиск
    """)
    col1, col2, col3 = st.columns(3)
    
    with col2:
        with open(films_paths[0], 'rb') as file:
            st.download_button(label='American Psycho',
        data=file, file_name='American Psycho 2000 1080p GER Blu-ray AVC DTS-HD MA 5.1-PEGASUS.srt')
    with col3:
        with open(films_paths[1], 'rb') as file:
            st.download_button(label='Breakfast at Tiffany\'s',
                           data=file, file_name='Breakfast at Tiffany\'s BDRip 1080p.eng.srt')
    with col1:
        with open(films_paths[2], 'rb') as file:
            st.download_button(label='The Lion King 1994 USA',
                           data=file, file_name='streamlit/The Lion King 1994 USA The Circle of Life Edition The Signature Collection 1080p Blu-ray AVC DTS-HD MA 7.1 - BluDragon_SDH.srt')

    sub_file = st.file_uploader(label='Загрузите файл формата srt', 
                                         type='srt')
    if sub_file is not None:
        sub_text = file_uploaded(sub_file=sub_file)
        tf_idf_loaded = vectorize(sub_text)
        prediction_proda = get_prediction(tf_idf_loaded)
        fig, ax = plt.subplots(1, figsize=(5,3))
        sns.barplot(x=english_levels, 
                    y=np.cumsum(prediction_proda),
                    ax=ax).set(title='Доля понятной лексики в фильме, \nв зависимости от уровня знания языка',
                               ylabel='Вероятность')
        for i, p in enumerate(ax.patches):
            text = np.cumsum(prediction_proda).round(2)
            ax.annotate(text[i], xy=(p.get_x() + p.get_width() / 2, 0.8 * p.get_height()), \
                    size=13, color='black', ha = 'center', va = 'center',
                    bbox = dict(boxstyle = 'round',\
                    facecolor='none',edgecolor='black', alpha = 0.5) )
        st.pyplot(fig, use_container_width=False)
        
def get_prediction(tf_idf_loaded):
    with open('streamlit/cbc_gpu_baseline.pickle', 'rb') as file:
        cbc_baseline = pickle.load(file)

    pred = cbc_baseline.predict_proba(tf_idf_loaded)
    return pred[0, :]

def vectorize(sub_text):
    with open('streamlit/tfidf.pkl', 'rb') as file:
        tfidf = pickle.load(file)
    tf_idf_loaded = tfidf.transform(np.array([sub_text]))
    return tf_idf_loaded
    

def file_uploaded(sub_file):
    sub_text = preprocess_text(str(sub_file.readlines()))
    sub_text = lemmatize(sub_text)
    return sub_text

def preprocess_text(data, stopwords=stopwords.words('english')):
    #Приводим к нижему регистру
    text = data.lower()
    text = re.sub('\n', '', text)
    #Удаляем все символы между <>, '<some symbols>' -> ''
    text = re.sub('<[^>]+>', '', text)
    #Добавляем пробелы между знаками препинания
    text = re.sub(r"([.,!?])", r" \1 ", text)
    # удаляем пробелы в начале и в конце предложения
    text = text.strip()
    #Удаляем слова в скобках
    text = re.sub(r'\([^)]*\)', '', text)
    # Оставляем только латинские буквы
    text = re.sub(r'[^a-z]', ' ', text)
    # удаление стоп-слов
    text = [w for w in text.split() if w not in stopwords]
    # удаляем слова короче 3х символов
    text = [w for w in text if len(w) >= 3]
    return ' '.join(text)

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])



def main():
    show_main_info()



if __name__ == "__main__":
    main()
