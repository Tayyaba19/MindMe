import os
import asyncio
import load_data
import streamlit as st
from lightrag.utils import setup_logger
from lightrag import QueryParam
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

setup_logger("streamlit", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# Streamlit UI
st.title("MindMe")

query_mode = "global"
user_query = st.text_input("Lets Talk, you first please!")

from graphshow import  show_graph 
from streamlit.components.v1 import iframe

import networkx as nx
from pyvis.network import Network
import random
import streamlit.components.v1 as components
   
if st.button("Show Knowledge Graph"):
    show_graph()
    path_to_html = "./knowledge_graph.html" 

    with open(path_to_html,'r') as f: 
        html_data = f.read()

    # Show in webpage
    st.header("Knowledge Graph")
    st.components.v1.html(html_data, height=300, width=700, scrolling=True)
    # st.components.v1.html(html_data)


async def initialize_rag():
    """Initialize LightRAG pipeline."""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

# Initialize RAG once and keep it in session
if "rag" not in st.session_state:
    st.session_state.rag = asyncio.run(initialize_rag())
    data = load_data.get_data()
    st.session_state.rag.insert(data)
    
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


async def get_async_response(query, query_param):
    return await st.session_state.rag.aquery(query, param=query_param)

if user_query:
    st.session_state.conversation_history.append({"role": "user", "content": user_query})
    query_param = QueryParam(
        mode=query_mode,
        conversation_history=st.session_state.conversation_history,
        history_turns=min(5, len(st.session_state.conversation_history)) 
    )
    
    # Await async query safely inside Streamlit
    response = asyncio.run(get_async_response(user_query, query_param))
    
    st.session_state.conversation_history.append({"role": "assistant", "content": response})
    st.write("### Answer:")
    st.write(response)

    
    
