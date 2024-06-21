import json
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
import threading

# Load environment variables
load_dotenv()

# Set up Google Generative AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_API_KEY")

# Function to extract text from the specified JSON file
def get_pdf_text(json_file):
    # Read JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract array of vendor descriptions
    vendor_descriptions = data.get('vendor_descriptions', [])

    # Concatenate all descriptions into a single string
    text = "\n".join(vendor_descriptions)
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

# Function to convert text to speech asynchronously
def text_to_speech_async(text):
    def run_speech_engine(text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    thread = threading.Thread(target=run_speech_engine, args=(text,))
    thread.start()

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
    st.session_state["gemini_response"] = gemini_response
    st.session_state["show_recommendation"] = True
    text_to_speech_async(gemini_response)  # Convert response to speech asynchronously

# Function to authenticate admin login
def authenticate(username, password):
    return username == "admin" and password == "12345"

# Function to load and edit JSON
def edit_json():
    json_data = {}

    # Load JSON data from file
    try:
        with open("test.json", "r") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        st.error("JSON file not found. Please make sure 'test.json' exists in the current directory.")
        return

    st.header("Edit JSON")
    st.write("Current JSON Content:")
    st.write(json_data)

    # Display text area for editing JSON content
    new_content = st.text_area("Edit JSON content:", value=json.dumps(json_data, indent=4))

    # Save button to update JSON file
    if st.button("Save JSON"):
        try:
            updated_data = json.loads(new_content)
            with open("test.json", "w") as f:
                json.dump(updated_data, f, indent=4)
            st.success("JSON updated successfully!")
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}")
        except Exception as e:
            st.error(f"Error saving JSON: {e}")
# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Admin Panel", layout="wide")

    # Add tabs for navigation
    tabs = ["Interaction", "Vendor Search", "Admin"]
    tab_choice = st.sidebar.radio("Navigation", tabs)

    if tab_choice == "Interaction":
        st.header("Ask me any question 💁")
        pdf_path = "test.json"
        raw_text = get_pdf_text(pdf_path)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("Database Indexed Successfully.")

        user_question = st.text_input("Ask a Question", key="user_question_text")
        if st.button("Speak a Question"):
            user_question = speech_to_text()  # Get input from speech
            if user_question:
                user_input(user_question)
        elif st.button("Submit", key="submit_question"):
            if user_question:
                user_input(user_question)
        
        if "gemini_response" in st.session_state:
            st.write("Gemini's Response: ", st.session_state["gemini_response"])

            if st.session_state.get("show_recommendation", False):
                recommend = st.radio("Do you want a recommendation based on Gemini's response?", ("Yes", "No"), key="recommend")
                if recommend == "Yes":
                    recommendation_query = st.text_input("What would you like a recommendation for? ask like this 'where can I buy'", key="recommendation_query")
                    if st.button("Get Recommendation", key="get_recommendation_button"):
                        if recommendation_query:
                            docs = search_pdf_for_vendors(recommendation_query)
                            st.write("Recommended Vendors:")
                            st.write("Reply: ", docs["output_text"])
                            text_to_speech_async(docs["output_text"])  # Convert recommendation to speech asynchronously
                            st.session_state["show_recommendation"] = False

    elif tab_choice == "Vendor Search":
        st.header("Vendor Search")

        pdf_path = "test.json"
        raw_text = get_pdf_text(pdf_path)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("Database Indexed Successfully.")

        user_query = st.text_input("Enter your query:", key="vendor_query")
        if st.button("Speak Query"):
            user_query = speech_to_text()  # Get input from speech
            if user_query:
                docs = search_pdf_for_vendors(user_query)
                if docs:
                    st.write("Recommended Vendors:")
                    st.write("Reply: ", docs["output_text"])
                    text_to_speech_async(docs["output_text"])  # Convert recommendation to speech asynchronously
        if st.button("Search"):
            if user_query:
                docs = search_pdf_for_vendors(user_query)
                if docs:
                    st.write("Recommended Vendors:")
                    st.write("Reply: ", docs["output_text"])
                    text_to_speech_async(docs["output_text"])  # Convert recommendation to speech asynchronously

    elif tab_choice == "Admin":
        st.header("Admin Section")

        # Check if already logged in
        if "admin_logged_in" not in st.session_state:
            # Login form
            st.subheader("Login to Admin Panel")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if authenticate(username, password):
                    st.session_state["admin_logged_in"] = True
                    st.success(f"Logged in as {username}")
                else:
                    st.error("Invalid username or password")
        else:
            edit_json()

    # Sidebar option to clear cache
    if st.sidebar.button("Clear Cache"):
        st.sidebar.success("Cache Cleared")

if __name__ == "__main__":
    main()