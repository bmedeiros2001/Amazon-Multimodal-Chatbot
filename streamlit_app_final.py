# streamlit_app_final.py

import streamlit as st
import os
from rag_functions_final import get_answer_for_text, get_answer_for_image, load_multimodal_vectorstore
from io import BytesIO
import pickle

#  Page configuration (MUST be the first command) 
st.set_page_config(
    page_title="E-commerce AI Assistant",
    layout="wide",
)

#  CSS Styling 
st.markdown("""
<style>
    .main-header {
        background-color: #800000;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1, .main-header p {
        color: white;
        text-align: center;
        margin: 0;
    }
    .main-header h1 { font-size: 2.5rem; }
    .main-header p { font-size: 1.2rem; margin-top: 0.5rem; color: #FFD700; }
    .st-emotion-cache-16txtl3 { padding-top: 2rem; } /* Sidebar padding */
    .st-emotion-cache-1jicfl2 { gap: 0.5rem; } /* Space between UI elements */
    .uploaded-image-preview {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 10px;
        border: 1px dashed #800000;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

#  Header 
st.markdown("""
<div class="main-header">
    <h1>E-commerce Product AI Assistant</h1>
    <p>Using Vision-Language Models for Customer Support</p>
</div>
""", unsafe_allow_html=True)

#  Load data and models 
vectorstore = load_multimodal_vectorstore()

#  Sidebar for API Key Input 
with st.sidebar:
    st.header("Configuration")
    if 'keys_set' in st.session_state and st.session_state.keys_set:
        st.success("✅ API Key is set.")
    else:
        st.subheader("Enter OpenAI API Key")
        openai_api_key_input = st.text_input("OpenAI API Key", type="password", label_visibility="collapsed")
        if st.button("Set Key"):
            if openai_api_key_input:
                os.environ["OPENAI_API_KEY"] = openai_api_key_input
                st.session_state.keys_set = True
                st.rerun()
            else:
                st.error("Please provide the OpenAI API key.")
    st.divider()
    st.subheader("Query by Image")
    uploaded_image = st.file_uploader(
        "Upload an image of a product to identify it.",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_image is not None:
        # Display a preview of the uploaded image
        st.image(uploaded_image, caption="Your Uploaded Image", width=200)

#  Main App Logic 
if vectorstore is None:
    st.error("Database could not be loaded. App cannot proceed.")
else:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("images"):
                st.image(message["images"], width=150, caption="Relevant Images")
            if message.get("sources"):
                st.info(f"Sources: {', '.join(message['sources'])}")

    #  Logic to handle image query 
    if uploaded_image is not None and "last_image_processed" not in st.session_state:
        if 'keys_set' not in st.session_state or not st.session_state.keys_set:
            st.warning("Please set your OpenAI API key before analyzing an image.")
        else:
            # Add a placeholder to chat history
            st.chat_message("user").markdown("*(Analyzing uploaded image...)*")
            st.session_state.messages.append({"role": "user", "content": f"*(Analyzed image: {uploaded_image.name})*"})

            # Get response from the image RAG function
            with st.spinner("Identifying product..."):
                # Convert the uploaded file to bytes
                image_bytes = BytesIO(uploaded_image.getvalue())
                response = get_answer_for_image(vectorstore, image_bytes, os.environ["OPENAI_API_KEY"])

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
                if response["images"]:
                    st.image(response["images"], width=150, caption="Similar Products Found")
                if response["sources"]:
                    st.info(f"Sources: {', '.join(response['sources'])}")

            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "images": response["images"],
                "sources": response["sources"]
            })
            # Use session state to prevent re-running the analysis on every interaction
            st.session_state.last_image_processed = True
            st.rerun() # Rerun to clear the "uploaded_image" state visually

    #  Logic to handle text query 
    if prompt := st.chat_input("Ask about a product..."):
        # Clear the image processing flag if user types a new question
        if "last_image_processed" in st.session_state:
            del st.session_state.last_image_processed
            
        if 'keys_set' not in st.session_state or not st.session_state.keys_set:
            st.warning("Please set your OpenAI API key first.")
        else:
            # Display user message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get response from the text RAG function
            with st.spinner("Searching..."):
                response = get_answer_for_text(vectorstore, prompt, os.environ["OPENAI_API_KEY"])

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
                if response["images"]:
                    st.image(response["images"], width=150, caption="Relevant Images")
                if response["sources"]:
                    st.info(f"Sources: {', '.join(response['sources'])}")

            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "images": response["images"],
                "sources": response["sources"]
            })