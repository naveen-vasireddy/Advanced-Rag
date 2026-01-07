from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Sample Data
docs = [
    Document(page_content="Climate change leads to rising sea levels due to melting ice caps."),
    Document(page_content="Higher global temperatures cause more frequent and intense heatwaves."),
    Document(page_content="Ocean acidification, a result of CO2 absorption, harms marine biodiversity.")
]

# Create Vector Store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    collection_name="climate_stats"
)

retriever = vectorstore.as_retriever()