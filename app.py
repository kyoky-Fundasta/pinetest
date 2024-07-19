import streamlit as st
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

import os

st.title("test pinecone")


def pinecone():
    st.write("pinecone starts")
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["Pinecone_API_KEY"]

    index_name = "fundasta"
    embeddings = OpenAIEmbeddings(
        openai_api_key=st.secrets["OPENAI_API_KEY"], model="text-embedding-3-small"
    )
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 6}
    )
    result = retriever.invoke("fundastaの育児休暇について?")
    print(result)
    st.write(result)


if st.button("click"):
    pinecone()
