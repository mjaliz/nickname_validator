import streamlit as st

from hatesonar import Sonar

st.title('Hate Speech detection')
query = st.text_input('Write your text')
sonar = Sonar()
if query != '':
    with st.spinner('Processing...'):
        result = sonar.ping(query)
        print(result)
        # if result == 1:
        #     st.error('Your text contains hate speech')
