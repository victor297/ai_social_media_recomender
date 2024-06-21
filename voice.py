import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pyttsx3
import speech_recognition as sr

# Load environment variables
load_dotenv()

# Set up Google Generative AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_API_KEY")

# Function to extract text from the specified PDF path
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question, make sure to provide all links to their profile and price also include details\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to get Gemini's response
def get_gemini_response(user_question):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    response = model.predict(user_question)
    return response

# Function to search PDF for vendors based on the user input
def search_pdf_for_vendors(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Set allow_dangerous_deserialization to True
    docs = new_db.similarity_search(user_query)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_query}, return_only_outputs=True)
    return response

# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio_data = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio_data)
            st.write("You said: ", text)
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I did not understand that.")
            return ""
        except sr.RequestError:
            st.write("Could not request results; check your network connection.")
            return ""

# Function to process user input and generate a response
def user_input(user_question):
    gemini_response = get_gemini_response(user_question)
    st.write("Gemini's Response: ", gemini_response)
    text_to_speech(gemini_response)  # Convert response to speech

    # Ask if the user wants a recommendation based on Gemini's response
    recommend = st.radio("Do you want a recommendation based on Gemini's response?", ("Yes", "No"))
    if recommend == "Yes":
        recommendation_query = st.text_input("What would you like a recommendation for? ask like this 'where can i buy' ")
        if st.button("Get Recommendation"):
            docs = search_pdf_for_vendors(recommendation_query)
            st.write("Recommended Vendors:")
            st.write("Reply: ", docs["output_text"])
            text_to_speech(docs["output_text"])  # Convert recommendation to speech

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Get Recommendations", layout="wide")
    
    # Add tabs for navigation
    tabs = ["Interaction", "Vendor Search"]
    tab_choice = st.sidebar.radio("Navigation", tabs)

    if tab_choice == "Interaction":
        st.header("Ask me any question 💁")
        pdf_path = "test.pdf"
        raw_text = get_pdf_text(pdf_path)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("Database Indexed Successfully.")

        user_question = st.text_input("Ask a Question")
        if st.button("Speak a Question"):
            user_question = speech_to_text()  # Get input from speech
        if user_question:
            user_input(user_question)

    elif tab_choice == "Vendor Search":
        st.header("Vendor Search")

        pdf_path = "test.pdf"
        raw_text = get_pdf_text(pdf_path)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("Database Indexed Successfully.")

        user_query = st.text_input("Enter your query:")
        if st.button("Speak Query"):
            user_query = speech_to_text()  # Get input from speech
        if st.button("Search"):
            if user_query:
                docs = search_pdf_for_vendors(user_query)
                if docs:
                   st.write("Recommended Vendors:")
                   st.write("Reply: ", docs["output_text"])
                   text_to_speech(docs["output_text"])  # Convert recommendation to speech

    # Sidebar option to clear cache
    if st.sidebar.button("Clear Cache"):
        os.remove("faiss_index")
        st.sidebar.success("Cache Cleared")

if __name__ == "__main__":
    main()
