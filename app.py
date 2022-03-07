import streamlit as st
import requests
import numpy as np

'''
# GenreGuesser
'''

st.write('This is a test of CI/CD blah blah blah')

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
    st.markdown('### Predicted Probabilities by Genre:')
    md_table = '| Genre | Probability |\n|----|----|'
    for genre, proba in labeled_probabilities:
        md_table += f'\n| {genre.capitalize()} | {round(proba, 3)} |'
    st.markdown(md_table)

lyrics = st.text_area(label = 'Lyrics to guess',
                      value = '',
                      height = 250)

if lyrics != '':
    params = {
        'lyrics' : lyrics,
    }
    gg_results = requests.get(url, params = params).json()
    write_prediction(gg_results)
    write_probabilities(gg_results)
