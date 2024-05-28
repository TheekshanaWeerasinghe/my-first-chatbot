import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI


OPENAI_API_KEY = "OpenAI_KEY"

# Upload PDF file
st.header("My first chatbot")
with st.sidebar:
    st.title("Your Document")
    file = st.file_uploader("Upload a pdf file and ask questions", type="pdf")

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    try:
        # Generating embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Creating vector store FAISS - Facebook AI semantic search
        vector_store = FAISS.from_texts(chunks, embeddings)

        # Get the user question
        user_question = st.text_input("Type your question here")

        # Do Similarity search
        if user_question:
            match = vector_store.similarity_search(user_question)
            st.write(match)

            # Define the llm
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=0,
                max_tokens=1000,
                model_name="gpt-3.5-turbo"
            )

            # Output result
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=match, question=user_question)
            st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
