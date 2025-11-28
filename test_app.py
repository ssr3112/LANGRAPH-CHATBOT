import streamlit as st

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'Hello! Type something.'}]

for msg in st.session_state['messages']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])

user_input = st.chat_input("Say something")

if user_input:
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
    st.session_state['messages'].append({'role': 'assistant', 'content': "Echo: " + user_input})
    with st.chat_message('assistant'):
        st.text("Echo: " + user_input)
