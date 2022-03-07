import streamlit as st
import requests

MODEL = 'svm'
MODEL_NAME = 'Support-Vector Machine'
GENRE_NAMES = ['country', 'pop', 'rock']

def list_genres(genre_list):
    output_string = ''
    for index, genre in enumerate(genre_list):
        output_string += f'**{genre}**'
        if index == len(GENRE_NAMES) - 2:
            output_string += ', and '
        elif index < len(GENRE_NAMES) - 2:
            output_string += ', '
    return output_string

st.title('GenreGuesser')

intro_text = f'''
                This app connects to a Google Cloud API hosting a {MODEL_NAME} model trained on thousands of songs
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
    if MODEL == 'svm':
        st.markdown('''
            _Note: Because of the way probabilities are calculated with `scikit-learn`'s Support Vector Machine models,
            there is a chance that the predicted genre is not the same as the genre with the highest assigned
            probability. See [here](https://scikit-learn.org/stable/modules/svm.html#scores-probabilities) for an explanation._
            ''')
    md_table = '| Genre | Probability |\n|----|----|'
    for genre, proba in labeled_probabilities:
        md_table += f'\n| {genre.capitalize()} | {round(proba * 100, 1)}% |'
    st.markdown(md_table)

lyrics = st.text_area(label = '',
                      value = '',
                      height = 250)

if lyrics != '':
    params = {
        'lyrics' : lyrics,
    }
    gg_results = requests.get(url, params = params).json()
    write_prediction(gg_results)
    write_probabilities(gg_results)
