import streamlit as st
from langchain_groq import ChatGroq
import os


def get_model():
    groq_api_key = st.secrets['groq_api_key']
    model_name = 'deepseek-r1-distill-llama-70b'

    groq_api_key = api_key
    os.environ['GROQ_API_KEY'] = groq_api_key

    # model_name = selected_model_name

    model = ChatGroq(
        model=model_name
    )
    return model 


def get_llm_models():
    models = {
        'Deepseek R1': 'deepseek-r1-distill-llama-70b',
    }
    return models


def generate_response(query, model, chat_history):
    # Chat History
    try:
        recent_history = chat_history[-15:]
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history])
    except:
        history_text = ""

    prompt = f'''
    Chat History: {history_text}

    New Query: {query}

    '''

    response = model.invoke(prompt)

    return response.content


def main():
    

    # Streamlit App
    st.title(":whale: Deepseek R1 Chat ")

    # Model Selection
    # selected_model = st.sidebar.selectbox("Select LLM Model", options=list(get_llm_models().keys()), index=0)

    # Get Model
    # models_dict = get_llm_models()
    model = get_model()

    print("Current Model:", selected_model)

    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input at the bottom
    user_question = st.chat_input("Ask Anything:")

    if user_question:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Display user question
        with st.chat_message("user", avatar="🙋‍♂️"):
            st.markdown(user_question)

        # Generate and display AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("LLM is Thinking..."):

                try:
                    response = generate_response(query=user_question, model=model, chat_history=st.session_state.chat_history)
                    
                    try:
                        thoughts = response.split('</think>')[0].strip().split('<think>')[-1].strip()
                        actual_response = response.split('</think>')[-1].strip()

                        final_response = f""" 
                        🤔Let Me Think:\n
                        {thoughts}

                        💬Response:\n
                        {actual_response}

                        """

                    except:
                        final_response = response

                    print(final_response)
                    
                    st.write(final_response)

                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                except:
                    st.error("Can't Find Any Solution, Please Try Again")



if __name__ == '__main__':
    main()
