# HYPERPARAMETER SUMMARY
## MMR prevents redundant results (important for e-commerce where you might have many similar products)
## k=5 balances context richness with token limits - too many docs can confuse the LLM

## temperature=0 for factual text queries (deterministic)
## temperature=0.1 for image analysis (slightly more creative interpretation)

import os
import streamlit as st
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import pickle

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

#  Custom CLIP Embedding Class 
@st.cache_resource
def load_clip_model_and_processor():
    # Loads CLIP model and processor.
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

class ClipEmbeddings(Embeddings):
    # Custom LangChain embedding class using raw transformers CLIP model.
    def __init__(self):
        self.model, self.processor = load_clip_model_and_processor()

    def _embed_text(self, texts):
        # Helper for embedding text.
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return embeddings.cpu().numpy()

    def embed_documents(self, texts):
        # Embeds documents for storage/retrieval.
        return self._embed_text(texts).tolist()

    def embed_query(self, text):
        # Embeds a single text query.
        return self._embed_text([text])[0].tolist()

    def embed_image(self, image_bytes):
        # Takes raw image bytes and returns a vector embedding.
        try:
            image = Image.open(image_bytes)
            inputs = self.processor(images=[image], return_tensors="pt")
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
            return embedding.cpu().numpy()[0].tolist()
        except Exception as e:
            st.error(f"Failed to embed image: {e}")
            return None

@st.cache_resource(show_spinner="Loading product database...")
def load_multimodal_vectorstore():
    try:
        embedding_function = ClipEmbeddings()
        vectorstore = Chroma(
            persist_directory="/Users/brunamedeiros/Documents/GitHub/Amazon-Multimodal-Chatbot/my_vectorstore_exploded_v2",
            embedding_function=embedding_function,
            collection_name="amazon_products_exploded_v2"
        )
        if vectorstore._collection.count() == 0:
            st.warning("Vectorstore is empty.")
            return None
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load vectorstore: {e}")
        return None

# def format_multimodal_docs(docs):
#     # Separates text and image URLs from retrieved docs.
#     text_snippets, image_urls = [], []
#     for doc in docs:
#         if doc.metadata.get("type") == "text":
#             text_snippets.append(doc.page_content)
#         elif doc.metadata.get("type") == "image":
#             if img_url := doc.page_content:
#                 image_urls.append(img_url)
#     return {"text_context": "\n\n".join(text_snippets), "image_urls": list(set(image_urls))}

def load_lookup_dictionaries():
    """Load the pickle files we created"""
    try:
        with open('image_lookup.pkl', 'rb') as f:
            image_lookup = pickle.load(f)
        with open('text_lookup.pkl', 'rb') as f:
            text_lookup = pickle.load(f)
        return image_lookup, text_lookup
    except Exception as e:
        print(f"Failed to load lookup dictionaries: {e}")
        return {}, {}

def format_multimodal_docs_with_lookups(docs):
    """SOLUTION: use lookup dictionaries instead of doc.page_content"""
    image_lookup, text_lookup = load_lookup_dictionaries()  # Load our phone books
    
    text_snippets = []
    image_urls = []

    for doc in docs:
        uniq_id = doc.metadata.get("uniq_id") # get product id from metadata
        doc_type = doc.metadata.get("type")
        
        if doc_type == "text" and uniq_id in text_lookup:
            # Get text chunk from lookup
            chunk_num = doc.metadata.get("chunk", 1)
            chunks = text_lookup[uniq_id]
            if chunk_num <= len(chunks):
                text_content = chunks[chunk_num - 1]
                text_snippets.append(text_content)  # ← Now this variable exists!

        elif doc_type == "image" and uniq_id in image_lookup:
            # Get ALL image URLs for this product
            product_image_urls = image_lookup[uniq_id]
            image_urls.extend(product_image_urls)  # ← Add all images, not just one
    
    return {
        "text_context": "\n\n".join(text_snippets), 
        "image_urls": list(set(image_urls))  # Remove duplicates
    }

def setup_rag_chain_for_text(vectorstore, openai_api_key):
    # Sets up RAG chain for text queries.
    if vectorstore is None: return None
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)

        # DICTIONARY SOLUTION: bypass broken vectorstore.as_retriever()
        def custom_retriever(question):
            # step 1: convert user question to embedding
            embedding_function = ClipEmbeddings()
            query_embedding = embedding_function.embed_query(question)

            # step 2: search database for similar items
            raw_results = vectorstore._collection.query(
                query_embeddings =[query_embedding],
                n_results = 5,
                include=['metadatas']
            )

            # step 3: convert results to format lookup function expects
            docs = []
            for metadata in raw_results['metadatas'][0]:
                fake_doc = type('FakeDoc', (), {'metadata':metadata})()
                docs.append(fake_doc)
            return docs

        #retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        # MMR = prevents reduntant reults (important for e-commerce where you might havee many similar products)
        # k=5 retrieves 5 docs


        prompt = hub.pull("rlm/rag-prompt")
        rag_chain_initial = RunnableParallel({"context": custom_retriever, "question": RunnablePassthrough()})
        answer_chain = (
            rag_chain_initial
            | (lambda x: {"context": format_multimodal_docs_with_lookups(x["context"])["text_context"], "question": x["question"]})
            | prompt | llm | StrOutputParser()
        )
        #return RunnableParallel({"answer": answer_chain, "context_docs": rag_chain_initial["context"]})
        return RunnableParallel({"answer": answer_chain, "context_docs": custom_retriever})
    except Exception as e:
        st.error(f"Failed to set up text RAG chain: {e}")
        return None

def get_answer_for_text(vectorstore, question, openai_api_key):
    # Main function for handling text-based questions.
    try:
        rag_chain = setup_rag_chain_for_text(vectorstore, openai_api_key)
        if rag_chain is None: return {"answer": "Chatbot is not ready.", "sources": [], "images": []}
        result = rag_chain.invoke(question) # this will crash because documents = None
        retrieved_docs = result["context_docs"]
        unique_text_sources, unique_image_urls = set(), set()
        for doc in retrieved_docs:
            uniq_id = doc.metadata.get("uniq_id", "Unknown")
            if doc.metadata.get("type") == "text":
                unique_text_sources.add(f"Product ID: {uniq_id}")
            elif doc.metadata.get("type") == "image" and doc.page_content:
                unique_image_urls.add(doc.page_content)
        return {"answer": result["answer"], "sources": list(unique_text_sources), "images": list(unique_image_urls)}
    except Exception as e:
        return {"answer": f"Internal error: {str(e)}", "sources": [], "images": []}

def get_answer_for_image(vectorstore, image_bytes, openai_api_key):
    # Handles an image-based query.
    if vectorstore is None: return {"answer": "Database not loaded.", "sources": [], "images": []}
    try:
        embedding_function = ClipEmbeddings()
        image_embedding = embedding_function.embed_image(image_bytes)
        if image_embedding is None: return {"answer": "Could not process image.", "sources": [], "images": []}

        #retrieved_docs = vectorstore.similarity_search_by_vector(embedding=image_embedding, k=5)
        # BYPASS: Use raw ChromaDB for image search
        raw_results = vectorstore._collection.query(
            query_embeddings=[image_embedding],
            n_results=5,
            include=['metadatas']
        )

        # Convert to fake docs for your lookup function
        retrieved_docs = []
        for metadata in raw_results['metadatas'][0]:
            fake_doc = type('FakeDoc', (), {'metadata': metadata})()
            retrieved_docs.append(fake_doc)

        formatted_context = format_multimodal_docs_with_lookups(retrieved_docs)
        text_context = formatted_context["text_context"]
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.1)
        prompt_template = "Based on the following information, identify and describe the product shown in a user's uploaded image.\n\nContext:\n{context}\n\nQuestion: What is the product in the image? Describe its key features."
        prompt = hub.pull("rlm/rag-prompt").from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": text_context})
        unique_text_sources, unique_image_urls = set(), set()
        for doc in retrieved_docs:
            uniq_id = doc.metadata.get("uniq_id", "Unknown")
            if doc.metadata.get("type") == "text":
                unique_text_sources.add(f"Product ID: {uniq_id}")
            elif doc.metadata.get("type") == "image" and doc.page_content:
                unique_image_urls.add(doc.page_content)
        return {"answer": answer, "sources": list(unique_text_sources), "images": list(unique_image_urls)}
    except Exception as e:
        return {"answer": f"Error during image analysis: {str(e)}", "sources": [], "images": []}