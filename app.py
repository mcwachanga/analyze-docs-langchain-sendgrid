import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from flask_mail import Mail, Message

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.sendgrid.net'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = "apikey"
app.config['MAIL_PASSWORD'] = os.environ.get('SENDGRID_API_KEY')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')
mail = Mail(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

llm=Cohere(model="command", temperature=0)
db = None

def extract_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    return docs

def embeddings(docs):
    embeddings = CohereEmbeddings(model = "multilingual-22-12")
    global db
    db = Chroma.from_documents(docs, embeddings)

def chat_pdf(query):
    matching_docs = db.similarity_search(query)
    chain = load_qa_chain(llm, chain_type="stuff",verbose=True)
    answer =  chain.run(input_documents=matching_docs, question=query)
    return answer

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "Upload a file!"
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        docs = extract_pdf(filepath)
        embeddings(docs)
        return "Done"
       
    return "Upload a valid file"

@app.route('/query', methods=['POST'])
def emails():
    query = request.json['query']
    email = request.json['email']
    answer = chat_pdf(query)
    msg = Message('PDF Chat', recipients=[email])
    msg.body = answer
    mail.send(msg)
    return "Email Sent"

if __name__ == '__main__':
    app.run(debug=True)

