import streamlit as st
import pickle

@st.cache_data
def get_tokenizer():
    with open('data/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        return tokenizer
    
@st.cache_data
def get_models():
    with open('data/Gradient Boosting.sav', 'rb') as boosting, \
      open('data/Naive Bayes.sav', 'rb') as bayes, \
      open('data/Random Forest.sav', 'rb') as forest:
        
        gboosting = pickle.load(boosting)
        nbayes = pickle.load(bayes)
        randForest = pickle.load(forest)
        return gboosting, nbayes, randForest
    
tokenizer = get_tokenizer()
gboosting, nbayes, randForest = get_models()

def predict(model, text):
    tokens = tokenizer.transform([text])
    prediction = model.predict(tokens)
    return prediction[0]



st.title('Определение тональности отзыва')

tab1, tab2, tab3 = st.tabs(["Бустинг","Случайный лес" ,"Вероятностная модель"])


with tab1:
    inp = st.text_area('Введите отзыв', key=0)
    if (inp):
        prediction = predict(gboosting, inp)
        st.write(f"Отзыв отнесен к категории {prediction}")

with tab2:
    inp = st.text_area('Введите отзыв',key=1)
    if (inp):
        prediction = predict(randForest, inp)
        st.write(f"Отзыв отнесен к категории {prediction}")

with tab3:
    inp = st.text_area('Введите отзыв',key=2)
    if (inp):
        prediction = predict(nbayes, inp)
        st.write(f"Отзыв отнесен к категории {prediction}")