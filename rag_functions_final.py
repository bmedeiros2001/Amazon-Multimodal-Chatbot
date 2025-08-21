# rag_functions_final.py

import os
import streamlit as st
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
import io
import re

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# --- CORE COMPONENTS (No changes needed) ---

@st.cache_resource
def load_clip_model_and_processor():
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

class ClipEmbeddings(Embeddings):
    def __init__(self):
        self.model, self.processor = load_clip_model_and_processor()
    def _embed_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return embeddings.cpu().numpy()
    def embed_documents(self, texts):
        return self._embed_text(texts).tolist()
    def embed_query(self, text):
        return self._embed_text([text])[0].tolist()
    def embed_image(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB": image = image.convert("RGB")
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
        vectorstore = Chroma(persist_directory="my_vectorstore_exploded_v3", embedding_function=embedding_function, collection_name="amazon_products_exploded_v3")
        if vectorstore._collection.count() == 0:
            st.warning("Vectorstore is empty.")
            return None, None
        return vectorstore, embedding_function
    except Exception as e:
        st.error(f"Failed to load vectorstore: {e}")
        return None, None

# --- ENHANCED RERANKING HELPER ---
def rerank_with_scores(query, docs):
    query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
    
    scored_docs = []
    for doc in docs:
        if not doc.page_content:
            score = 0
        else:
            doc_words = set(re.findall(r'\b\w+\b', doc.page_content.lower()))
            score = len(query_keywords.intersection(doc_words))
        scored_docs.append({"doc": doc, "score": score})

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    
    best_score = scored_docs[0]["score"] if scored_docs else 0
    sorted_docs = [item["doc"] for item in scored_docs]
    
    return sorted_docs, best_score

# --- ROBUST RAG FUNCTIONS (DEFINITIVE FINAL VERSION) ---

def get_answer_for_text(vectorstore, embedding_function, question, openai_api_key):
    # This function implements the robust RAG with Fallback logic.
    MINIMUM_RELEVANCE_SCORE = 2

    try:
        query_embedding = embedding_function.embed_query(question)
        results = vectorstore._collection.query(
            query_embeddings=[query_embedding],
            n_results=25,
            where={"type": "text"}
        )
        candidate_docs = [Document(page_content=content, metadata=metadata) for content, metadata in zip(results["documents"][0], results["metadatas"][0]) if content]

        reranked_docs, best_score = rerank_with_scores(question, candidate_docs)

        if best_score >= MINIMUM_RELEVANCE_SCORE:
            st.info("Found relevant information in the product database...")
            top_docs = reranked_docs[:5]
            context_text = "\n\n".join(doc.page_content for doc in top_docs)
            
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)
            prompt = hub.pull("rlm/rag-prompt")
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context_text, "question": question})
            
            sources = list(set(f"Product ID: {doc.metadata.get('uniq_id', 'Unknown')}" for doc in top_docs))
            return {"answer": answer, "sources": sources, "images": []}
        else:
            st.warning("Could not find a specific match in the product database. Answering from general knowledge...")
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.1)
            prompt_template = """You are a helpful e-commerce AI assistant. You were unable to find information in the specific product database. Answer the user's question based on your own general knowledge. Preface your answer with a phrase like "I couldn't find specific information about that in our product database, but based on my general knowledge..."

User's Question: {question}"""
            prompt = PromptTemplate.from_template(prompt_template)
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"question": question})
            
            return {"answer": answer, "sources": ["General Knowledge"], "images": []}
    except Exception as e:
        return {"answer": f"Internal error: {str(e)}", "sources": [], "images": []}

def get_answer_for_image_and_text(vectorstore, embedding_function, image_bytes, question, openai_api_key):
    # --- THIS FUNCTION HAS BEEN RESTORED AS REQUESTED ---
    try:
        image_embedding = embedding_function.embed_image(image_bytes)
        if image_embedding is None: return {"answer": "Could not process the staged image.", "sources": [], "images": []}
        
        initial_results = vectorstore._collection.query(query_embeddings=[image_embedding], n_results=1)
        if not initial_results or not initial_results["ids"][0]:
            return {"answer": "I could not identify the product in the image.", "sources": [], "images": []}

        product_id = initial_results["metadatas"][0][0].get("uniq_id")
        if not product_id: return {"answer": "Found a match but it was missing an ID.", "sources": [], "images": []}

        text_results = vectorstore._collection.get(where={"$and": [{"uniq_id": product_id}, {"type": "text"}]})
        
        context_text = "\n\n".join(doc for doc in text_results['documents'] if doc)
        if not context_text.strip():
            return {"answer": f"I identified the product (ID: {product_id}) but could not find any descriptive details to answer your question.", "sources": [f"Product ID: {product_id}"], "images": []}

        prompt_template = "Based on the following information about the product in the user's image, please answer their specific question.\n\nUser's Question: {question}\n\nContext:\n{context}"
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context_text, "question": question})
        
        return {"answer": answer, "sources": [f"Product ID: {product_id}"], "images": []}
    except Exception as e:
        return {"answer": f"Error during combined analysis: {str(e)}", "sources": [], "images": []}


def get_image_by_text(vectorstore, embedding_function, question, openai_api_key):
    # --- THIS FUNCTION HAS BEEN RESTORED AS REQUESTED ---
    try:
        query_embedding = embedding_function.embed_query(question)
        results = vectorstore._collection.query(
            query_embeddings=[query_embedding],
            n_results=25,
            where={"type": "text"}
        )

        candidate_docs = [Document(page_content=content, metadata=metadata) for content, metadata in zip(results["documents"][0], results["metadatas"][0]) if content]
        # Using rerank_with_scores to just get the sorted docs
        reranked_docs, _ = rerank_with_scores(question, candidate_docs)

        if not reranked_docs:
            return {"answer": "I'm sorry, I couldn't find any products matching that description.", "sources": [], "images": []}
        
        best_match_doc = reranked_docs[0]
        product_id = best_match_doc.metadata.get("uniq_id")
        if not product_id: return {"answer": "Found a text match but it was missing an ID.", "sources": [], "images": []}

        image_results = vectorstore._collection.get(where={"$and": [{"uniq_id": product_id}, {"type": "image"}]}, limit=1)
        if not image_results or not image_results.get("documents") or not image_results["documents"][0]:
            return {"answer": f"I found the product (ID: {product_id}), but no image for it exists in the database.", "sources": [f"Product ID: {product_id}"], "images": []}

        image_url = image_results["documents"][0]
        answer = f"Sure, here is an image of the product you asked about:"
        return {"answer": answer, "sources": [f"Product ID: {product_id}"], "images": [image_url]}
    except Exception as e:
        return {"answer": f"An error occurred while finding an image: {str(e)}", "sources": [], "images": []}