import streamlit as st
import joblib
import pandas as pd

# For text-cleaning
import re, string
import nltk
nltk.download('stopwords', quiet = True)
nltk.download('punkt', quiet = True)
nltk.download('wordnet', quiet = True)
nltk.download('omw-1.4', quiet = True)
nltk.download('words', quiet = True)
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

MODEL = 'svm'
MODEL_NAME = 'Support-Vector Machine'
GENRE_NAMES = ['country', 'pop', 'rap', 'rock']

model = joblib.load('heroku.joblib')

def list_genres(genre_list):
    output_string = ''
    for index, genre in enumerate(genre_list):
        output_string += f'**{genre}**'
        if index == len(GENRE_NAMES) - 2:
            output_string += ', and '
        elif index < len(GENRE_NAMES) - 2:
            output_string += ', '
    return output_string

st.title('_GenreGuesser_')

intro_text = f'''
                This app connects to a cloud-hosted machine-learning model trained on the lyrics of thousands of songs
                classified into {len(GENRE_NAMES)} genres: {list_genres(GENRE_NAMES)}. In the text box below,
                put in the lyrics for a song from one of these genres, and see if we guess the genre correctly!
                '''

st.markdown(intro_text)

url = 'https://genre-guesser-2cfzxdapea-ew.a.run.app/predict_svm'
def write_prediction(input_results):
    predicted_genre = input_results['genre']
    st.markdown(f'### Predicted Genre: <span style = "color: green">{predicted_genre.capitalize()}</span>', unsafe_allow_html = True)

def write_probabilities(input_results):
    pred_proba = input_results['proba']
    genres = list(pred_proba.keys())
    probabilities = [pred_proba[genre] for genre in genres]
    labeled_probabilities = list(zip(genres, probabilities))
    labeled_probabilities.sort(reverse = True, key = lambda x : x[1])
    st.markdown('### Probability of Each Genre:')
    md_table = '| Genre | Probability |\n|----|----|'
    for genre, proba in labeled_probabilities:
        md_table += f'\n| {genre.capitalize()} | {round(proba * 100, 1)}% |'
    st.markdown(md_table)
    if MODEL == 'svm':
        st.markdown('''
            &nbsp;

            _Note: Because of the way probabilities are calculated with this particular model,
            there is a chance that the predicted genre is not the same as the genre with the highest assigned
            probability. See [here](https://scikit-learn.org/stable/modules/svm.html#scores-probabilities) for an explanation._
            ''')

lyrics = st.text_area(label = 'Enter lyrics here, then press Ctrl + Enter:',
                      value = '',
                      height = 250)

#if lyrics != '':
#    params = {
#        'lyrics' : lyrics,
#    }
#    gg_results = requests.get(url, params = params).json()
#    write_prediction(gg_results)
#    write_probabilities(gg_results)

def clean_text(text):
    #remove 'е'
    text = text.replace('е', 'e')

    #remove headers like [Chorus] etc
    headers = re.findall(r"\[(.*?)\]", text)
    for header in headers:
        text = text.replace(f'[{header}]', ' ')

    #separate lower/upper case words (like 'needHow')
    cap_sep_find = r'([a-z])([A-Z])'
    cap_sep_replace = r'\1 \2'
    text = re.sub(cap_sep_find, cap_sep_replace, text)

    #remove punctuation
    exclude = string.punctuation + "’‘”“"
    for punctuation in exclude:
           text = text.replace(punctuation, ' ')

    #turn text into lowercase
    text = text.lower()

    #remove numericals
    text = ''.join(word for word in text if not word.isdigit())

    #remove stopwords
    stop_words = set(stopwords.words('english'))

    #tokenise
    word_tokens = word_tokenize(text)
    text = [w for w in word_tokens if not w in stop_words]

    #lemmatise
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in text]
    text = lemmatized

    #filter out non-ascii words
    words_set = set(words.words())
    safe_set = set(['cliché', 'rosé', 'déjà', 'ménage',  'yoncé', 'beyoncé', 'café', 'crème', 'señor', 'señorita'])
    ascii_list = []
    for word in text:
        if word in words_set or word.isascii() or word in safe_set:
            ascii_list.append(word)
    text = ' '.join(ascii_list)

    text = text.replace('wan na', "wanna")
    text = text.replace('gon na', "gonna")
    text = text.replace('got ta', "gotta")

    return text

if lyrics!= '':
    test_lyrics = clean_text(lyrics)
    predicted_genre = model.predict(pd.Series([test_lyrics]))[0]
    predicted_probas = model.predict_proba(pd.Series([test_lyrics]))
    proba_classes = model.classes_
    output_dict = {}
    for index, genre in enumerate(proba_classes):
        output_dict[genre] = predicted_probas[0,index]
    gg_results = {'genre' : predicted_genre,
                  'proba' : output_dict}
    write_prediction(gg_results)
    write_probabilities(gg_results)
