import streamlit as st
from streamlit_backend import chatbot  # Your WORKING backend
from langchain_core.messages import HumanMessage
import time

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Show history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    # User message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # Assistant with TYPING EFFECT
    assistant_container = st.chat_message('assistant')
    with assistant_container:
        # Typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown("**🤖 AI is typing...**")
        
        # Get full response from your WORKING backend
        response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
        full_response = response['messages'][-1].content
        
        # SIMULATE typing by revealing text gradually
        typing_placeholder.markdown("")
        for i in range(0, len(full_response), 2):
            partial_response = full_response[:i+2]
            typing_placeholder.markdown(partial_response)
            time.sleep(0.03)  # Typing speed
            
        # Final full response
        typing_placeholder.markdown(full_response)

    # Save to history
    st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})
