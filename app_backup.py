import os
import sqlite3
import traceback

from flask import (
    Flask,
    render_template,
    render_template_string,
    request,
    session,
    redirect,
    url_for
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

from config import (
    EMBEDDING_MODEL,
    PINECONE_INDEX,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from embeddings.sentence_transformer import SentenceEmbedding
from llms.groq_llama import GroqLLM
from pipeline import RAGPipeline
from vectorstores.pinecone_store import PineconeStore
from loaders.pdf_loader import PDFLoader
from splitters.text_splitter import TextSplitter


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

# Change this in production
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "change-this-secret-key"
)

UPLOAD_FOLDER = "data"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# GLOBAL RAG OBJECTS
# ============================================================

# These are loaded only when required.
# This prevents Flask from failing during startup if
# Pinecone/model initialization has a problem.

embedding = None
vectorstore = None


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():
    """
    Create a SQLite database connection.
    """
    return sqlite3.connect("users.db")


def init_db():
    """
    Create required database tables.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    # -------------------------
    # USERS TABLE
    # -------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # -------------------------
    # DOCUMENTS TABLE
    # -------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            version TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Processing'
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# HELPER - UPDATE DOCUMENT STATUS
# ============================================================

def update_document_status(document_id, status):
    """
    Update document processing status.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE documents
        SET status = ?
        WHERE id = ?
    """, (status, document_id))

    conn.commit()
    conn.close()


# ============================================================
# HELPER - GET DOCUMENT
# ============================================================

def get_document(document_id):
    """
    Get document information from SQLite.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, filename, version, status
        FROM documents
        WHERE id = ?
    """, (document_id,))

    document = cursor.fetchone()

    conn.close()

    return document


# ============================================================
# LOGIN REQUIRED HELPER
# ============================================================

def login_required():
    """
    Check whether a user is logged in.
    """

    return "user_id" in session


# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not username or not password:
            return "Username and password are required."

        hashed_password = generate_password_hash(
            password
        )

        try:

            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO users (
                    username,
                    password
                )
                VALUES (?, ?)
            """, (
                username,
                hashed_password
            ))

            conn.commit()
            conn.close()

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:

            return "Username already exists."

    return render_template("register.html")


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, username, password
            FROM users
            WHERE username = ?
        """, (username,))

        user = cursor.fetchone()

        conn.close()

        if user and check_password_hash(
            user[2],
            password
        ):

            session["user_id"] = user[0]
            session["username"] = user[1]

            return redirect(
                url_for("documents")
            )

        return "Invalid username or password."

    return render_template("login.html")


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# DOCUMENT MANAGEMENT PAGE
# ============================================================

@app.route("/documents")
def documents():

    if not login_required():
        return redirect(
            url_for("login")
        )

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, filename, version, status
        FROM documents
        ORDER BY id DESC
    """)

    documents = cursor.fetchall()

    conn.close()

    return render_template(
        "documents.html",
        documents=documents
    )


# ============================================================
# RAG INITIALIZATION
# ============================================================

def initialize_embedding():
    """
    Load embedding model only when needed.
    """

    global embedding

    if embedding is None:

        print()
        print("=" * 60)
        print("LOADING EMBEDDING MODEL")
        print("=" * 60)

        print(
            f"Embedding model: {EMBEDDING_MODEL}"
        )

        embedding = (
            SentenceEmbedding(
                EMBEDDING_MODEL
            ).get_model()
        )

        print(
            "Embedding model loaded successfully."
        )

    return embedding


def initialize_vectorstore():
    """
    Connect to the existing Pinecone index.
    """

    global vectorstore

    if vectorstore is None:

        embedding_model = (
            initialize_embedding()
        )

        print()
        print("=" * 60)
        print("CONNECTING TO PINECONE")
        print("=" * 60)

        print(
            f"Pinecone index: {PINECONE_INDEX}"
        )

        vectorstore = PineconeStore(
            PINECONE_INDEX,
            embedding_model
        ).connect()

        print(
            "Pinecone connection successful."
        )

    return vectorstore


# ============================================================
# PDF INGESTION
# ============================================================

def ingest_pdf(
    pdf_path,
    document_id,
    filename
):
    """
    Complete PDF ingestion pipeline:

        PDF
         ↓
        Load
         ↓
        Split
         ↓
        Add metadata
         ↓
        Create embeddings
         ↓
        Store in Pinecone
         ↓
        Return chunk count
    """

    print()
    print("=" * 70)
    print("STARTING PDF INGESTION")
    print("=" * 70)

    print(f"File       : {filename}")
    print(f"Document ID: {document_id}")
    print(f"File path  : {pdf_path}")

    # --------------------------------------------------------
    # STEP 1 - LOAD PDF
    # --------------------------------------------------------

    print()
    print("[1/5] LOADING PDF")
    print("-" * 70)

    pdf_loader = PDFLoader()

    loaded_documents = (
        pdf_loader.load(pdf_path)
    )

    if not loaded_documents:
        raise Exception(
            "PDF loader returned no content."
        )

    print(
        f"Loaded {len(loaded_documents)} document section(s)."
    )

    # --------------------------------------------------------
    # STEP 2 - SPLIT INTO CHUNKS
    # --------------------------------------------------------

    print()
    print("[2/5] SPLITTING TEXT INTO CHUNKS")
    print("-" * 70)

    print(
        f"Chunk size    : {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap : {CHUNK_OVERLAP}"
    )

    text_splitter = TextSplitter(
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    documents = text_splitter.split(
        loaded_documents
    )

    if not documents:
        raise Exception(
            "Text splitter created zero chunks."
        )

    print(
        f"Created {len(documents)} chunks."
    )

    # --------------------------------------------------------
    # STEP 3 - SHOW CHUNKS + ADD METADATA
    # --------------------------------------------------------

    print()
    print("[3/5] PREPARING CHUNKS")
    print("-" * 70)

    print(
        "Adding document metadata to every chunk..."
    )

    for index, document in enumerate(
        documents,
        start=1
    ):

        # Make sure metadata exists
        if not hasattr(
            document,
            "metadata"
        ):
            document.metadata = {}

        # IMPORTANT:
        # This metadata is later used to make
        # retrieval document-specific.

        document.metadata[
            "document_id"
        ] = str(document_id)

        document.metadata[
            "filename"
        ] = filename

        print()
        print("-" * 60)
        print(f"CHUNK {index}")
        print("-" * 60)

        print(
            document.page_content
        )

        print()
        print(
            "Metadata:",
            document.metadata
        )

    print()
    print(
        f"Prepared {len(documents)} chunks."
    )

    # --------------------------------------------------------
    # STEP 4 - LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    print()
    print("[4/5] CREATING EMBEDDINGS")
    print("-" * 70)

    embedding_model = (
        initialize_embedding()
    )

    print(
        "Embedding model is ready."
    )

    # --------------------------------------------------------
    # STEP 5 - STORE IN PINECONE
    # --------------------------------------------------------

    print()
    print("[5/5] STORING CHUNKS IN PINECONE")
    print("-" * 70)

    print(
        f"Index: {PINECONE_INDEX}"
    )

    store = PineconeStore(
        PINECONE_INDEX,
        embedding_model
    )

    store.create_store(
        documents
    )

    print()
    print(
        f"Successfully stored {len(documents)} chunks in Pinecone."
    )

    print()
    print("=" * 70)
    print("PDF INGESTION SUCCESSFUL")
    print("=" * 70)

    print(
        f"Document ID : {document_id}"
    )

    print(
        f"Filename    : {filename}"
    )

    print(
        f"Chunks      : {len(documents)}"
    )

    print(
        "Status      : Approved"
    )

    print("=" * 70)

    return len(documents)


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@app.route(
    "/upload-document",
    methods=["POST"]
)
def upload_document():

    if not login_required():
        return redirect(
            url_for("login")
        )

    # --------------------------------------------------------
    # GET FILE
    # --------------------------------------------------------

    file = request.files.get(
        "document"
    )

    if not file or file.filename == "":
        return "Please select a PDF document."

    # --------------------------------------------------------
    # SECURE FILE NAME
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )

    if not filename:
        return "Invalid filename."

    # --------------------------------------------------------
    # CHECK EXTENSION
    # --------------------------------------------------------

    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension != ".pdf":

        return (
            "Currently only PDF files "
            "are supported for RAG ingestion."
        )

    # --------------------------------------------------------
    # SAVE PDF
    # --------------------------------------------------------

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # If a file with the same name already exists,
    # create a unique filename.

    if os.path.exists(filepath):

        base_name = os.path.splitext(
            filename
        )[0]

        extension = os.path.splitext(
            filename
        )[1]

        counter = 1

        while os.path.exists(filepath):

            new_filename = (
                f"{base_name}_{counter}"
                f"{extension}"
            )

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                new_filename
            )

            counter += 1

        filename = new_filename

    file.save(filepath)

    print()
    print("=" * 70)
    print("PDF UPLOAD")
    print("=" * 70)

    print(
        f"Saved file: {filepath}"
    )

    # --------------------------------------------------------
    # INSERT INTO DATABASE
    # --------------------------------------------------------

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO documents (
            filename,
            version,
            status
        )
        VALUES (?, ?, ?)
    """, (
        filename,
        "v1.0",
        "Processing"
    ))

    document_id = cursor.lastrowid

    conn.commit()
    conn.close()

    print(
        f"Document ID: {document_id}"
    )

    print(
        "Status: Processing"
    )

    # --------------------------------------------------------
    # INGEST PDF
    # --------------------------------------------------------

    try:

        chunks_count = ingest_pdf(
            filepath,
            document_id,
            filename
        )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        update_document_status(
            document_id,
            "Approved"
        )

        print()
        print("=" * 70)
        print("DOCUMENT APPROVED")
        print("=" * 70)

        print(
            f"Document : {filename}"
        )

        print(
            f"Chunks   : {chunks_count}"
        )

        print(
            "Status   : Approved"
        )

        print("=" * 70)

    except Exception as e:

        # ----------------------------------------------------
        # INGESTION FAILED
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("INGESTION FAILED")
        print("=" * 70)

        print(
            f"Document : {filename}"
        )

        print(
            f"Document ID: {document_id}"
        )

        print(
            f"Error: {str(e)}"
        )

        print()
        print("FULL TRACEBACK:")
        traceback.print_exc()

        print("=" * 70)

        update_document_status(
            document_id,
            "Failed"
        )

        return (
            f"""
            <html>
            <head>
                <title>Document Processing Failed</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background: #f5f7fb;
                        padding: 40px;
                    }}

                    .box {{
                        max-width: 800px;
                        margin: auto;
                        background: white;
                        padding: 30px;
                        border-radius: 12px;
                        box-shadow:
                            0 5px 20px rgba(0,0,0,0.08);
                    }}

                    h2 {{
                        color: #c62828;
                    }}

                    .error {{
                        background: #fff1f1;
                        border-left: 4px solid #e53935;
                        padding: 15px;
                        margin-top: 20px;
                        white-space: pre-wrap;
                        word-break: break-word;
                    }}

                    a {{
                        display: inline-block;
                        margin-top: 20px;
                        padding: 10px 18px;
                        background: #6366f1;
                        color: white;
                        text-decoration: none;
                        border-radius: 7px;
                    }}
                </style>
            </head>

            <body>

                <div class="box">

                    <h2>
                        Document Processing Failed
                    </h2>

                    <p>
                        The PDF was uploaded, but the
                        RAG ingestion process failed.
                    </p>

                    <p>
                        <strong>Document:</strong>
                        {filename}
                    </p>

                    <p>
                        <strong>Document ID:</strong>
                        {document_id}
                    </p>

                    <div class="error">
                        {str(e)}
                    </div>

                    <a href="{url_for('documents')}">
                        Back to Documents
                    </a>

                </div>

            </body>
            </html>
            """
        )

    # --------------------------------------------------------
    # SUCCESS → DOCUMENTS PAGE
    # --------------------------------------------------------

    return redirect(
        url_for("documents")
    )


# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.route(
    "/delete-document/<int:document_id>",
    methods=["POST"]
)
def delete_document(document_id):

    if not login_required():
        return redirect(
            url_for("login")
        )

    document = get_document(
        document_id
    )

    if not document:
        return redirect(
            url_for("documents")
        )

    filename = document[1]

    # --------------------------------------------------------
    # DELETE LOCAL PDF
    # --------------------------------------------------------

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if os.path.exists(filepath):

        try:
            os.remove(filepath)

            print()
            print(
                f"Deleted local PDF: {filepath}"
            )

        except Exception as e:

            print(
                f"Could not delete local PDF: {e}"
            )

    # --------------------------------------------------------
    # DELETE DATABASE RECORD
    # --------------------------------------------------------

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM documents
        WHERE id = ?
    """, (document_id,))

    conn.commit()
    conn.close()

    print(
        f"Deleted database record: {document_id}"
    )

    # NOTE:
    # The Pinecone vectors are not deleted here.
    # Because document_id is unique, those old vectors
    # won't be selected for a different document.
    #
    # If you want true Pinecone vector deletion later,
    # we can add that separately.

    return redirect(
        url_for("documents")
    )


# ============================================================
# DOCUMENT-SPECIFIC RETRIEVER
# ============================================================

class DocumentRetriever:

    def __init__(
        self,
        vectorstore,
        document_id
    ):

        self.document_id = str(
            document_id
        )

        print()
        print(
            "=" * 60
        )

        print(
            "CREATING DOCUMENT-SPECIFIC RETRIEVER"
        )

        print(
            f"Document ID: {self.document_id}"
        )

        print(
            "=" * 60
        )

        self.retriever = (
            vectorstore.as_retriever(
                search_kwargs={
                    "k": TOP_K,

                    "filter": {
                        "document_id":
                            self.document_id
                    }
                }
            )
        )

    def retrieve(self, question):

        print()
        print(
            "RETRIEVING DOCUMENT CHUNKS"
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Document filter: {self.document_id}"
        )

        documents = (
            self.retriever.invoke(
                question
            )
        )

        print()
        print(
            f"Retrieved {len(documents)} chunk(s)."
        )

        return documents


# ============================================================
# ASKWISE HTML
# ============================================================

PAGE = """
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>Askwise</title>

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family:
                Arial,
                Helvetica,
                sans-serif;

            background:
                linear-gradient(
                    135deg,
                    #f5f7ff,
                    #ffffff
                );

            min-height: 100vh;

            color: #172033;
        }

        .navbar {
            height: 70px;

            display: flex;
            align-items: center;
            justify-content: space-between;

            padding: 0 42px;

            background: white;

            border-bottom:
                1px solid #e8eaf4;
        }

        .logo {
            font-size: 25px;
            font-weight: 800;
            color: #111827;
        }

        .logo span {
            color: #6366f1;
        }

        .nav-right {
            display: flex;
            gap: 20px;
            align-items: center;
        }

        .username {
            color: #6b7280;
            font-size: 14px;
        }

        .nav-link {
            text-decoration: none;
            color: #6366f1;
            font-weight: 600;
        }

        .container {
            max-width: 1100px;
            margin: 45px auto;
            padding: 0 20px;
        }

        .card {
            background: white;
            border-radius: 20px;
            padding: 38px;
            border: 1px solid #e4e7f2;

            box-shadow:
                0 10px 35px
                rgba(35, 45, 90, 0.08);
        }

        h1 {
            margin-top: 0;
            margin-bottom: 10px;

            font-size: 28px;
        }

        .subtitle {
            color: #7b8192;
            margin-bottom: 32px;
        }

        .document-info {
            background: #f5f6ff;
            border: 1px solid #e2e4ff;

            border-radius: 12px;

            padding: 18px 20px;

            margin-bottom: 28px;
        }

        .document-name {
            font-weight: 700;
            color: #374151;
        }

        .document-meta {
            margin-top: 8px;
            color: #777f92;
            font-size: 14px;
        }

        .status {
            display: inline-block;

            margin-left: 10px;

            padding: 5px 10px;

            border-radius: 20px;

            font-size: 12px;

            font-weight: 700;
        }

        .approved {
            background: #dcfce7;
            color: #166534;
        }

        .processing {
            background: #fef3c7;
            color: #92400e;
        }

        .failed {
            background: #fee2e2;
            color: #991b1b;
        }

        form {
            display: flex;
            gap: 12px;
        }

        input[type="text"] {
            flex: 1;

            height: 58px;

            border: 1px solid #d7dbea;

            border-radius: 12px;

            padding: 0 20px;

            font-size: 16px;

            outline: none;
        }

        input[type="text"]:focus {
            border-color: #6366f1;

            box-shadow:
                0 0 0 3px
                rgba(99, 102, 241, 0.1);
        }

        button {
            height: 58px;

            border: none;

            border-radius: 12px;

            padding: 0 32px;

            background:
                linear-gradient(
                    135deg,
                    #6366f1,
                    #5b7cfa
                );

            color: white;

            font-size: 16px;

            font-weight: 700;

            cursor: pointer;
        }

        button:hover {
            opacity: 0.93;
        }

        .answer {
            margin-top: 30px;

            background: #f8f9ff;

            border: 1px solid #e1e4f6;

            border-radius: 14px;

            padding: 25px;

            line-height: 1.7;

            white-space: pre-wrap;
        }

        .answer-title {
            font-weight: 800;
            color: #3730a3;
            margin-bottom: 12px;
        }

        .error {
            margin-top: 25px;

            background: #fff1f1;

            border-left: 4px solid #ef4444;

            border-radius: 8px;

            padding: 18px;

            color: #b91c1c;
        }

        .back {
            display: inline-block;

            margin-top: 25px;

            text-decoration: none;

            color: #6366f1;

            font-weight: 600;
        }

        @media (max-width: 700px) {

            .navbar {
                padding: 0 20px;
            }

            .container {
                margin: 25px auto;
            }

            .card {
                padding: 24px;
            }

            form {
                flex-direction: column;
            }

            button {
                width: 100%;
            }

            .username {
                display: none;
            }
        }

    </style>

</head>


<body>


<nav class="navbar">

    <div class="logo">
        Ask<span>wise</span>
    </div>

    <div class="nav-right">

        <span class="username">
            {{ session.get("username") }}
        </span>

        <a
            href="{{ url_for('documents') }}"
            class="nav-link"
        >
            Documents
        </a>

        <a
            href="{{ url_for('logout') }}"
            class="nav-link"
        >
            Logout
        </a>

    </div>

</nav>


<div class="container">

    <div class="card">

        <h1>
            Ask a Question
        </h1>

        {% if selected_document %}

            <div class="subtitle">

                Ask anything about
                <strong>
                    {{ selected_document[1] }}
                </strong>

            </div>


            <div class="document-info">

                <div class="document-name">

                    {{ selected_document[1] }}

                    {% if selected_document[3] == "Approved" %}

                        <span class="status approved">
                            Approved
                        </span>

                    {% elif selected_document[3] == "Processing" %}

                        <span class="status processing">
                            Processing
                        </span>

                    {% elif selected_document[3] == "Failed" %}

                        <span class="status failed">
                            Failed
                        </span>

                    {% endif %}

                </div>


                <div class="document-meta">

                    Version:
                    {{ selected_document[2] }}

                    &nbsp;&nbsp; | &nbsp;&nbsp;

                    Document ID:
                    {{ selected_document[0] }}

                </div>

            </div>


            {% if selected_document[3] == "Approved" %}

                <form
                    method="POST"
                    action="{{ url_for('home') }}"
                >

                    <input
                        type="hidden"
                        name="document_id"
                        value="{{ selected_document[0] }}"
                    >

                    <input
                        type="text"
                        name="question"
                        placeholder="Ask a question about this document..."
                        required
                    >

                    <button type="submit">
                        Ask
                    </button>

                </form>


            {% elif selected_document[3] == "Processing" %}

                <div class="error"
                     style="
                        background:#fffaf0;
                        color:#92400e;
                        border-left-color:#f59e0b;
                     ">

                    This document is still being processed.

                    Please wait until the status becomes
                    <strong>Approved</strong>.

                </div>


            {% elif selected_document[3] == "Failed" %}

                <div class="error">

                    Document ingestion failed.

                    Please upload the PDF again after
                    checking the Flask terminal for the
                    ingestion error.

                </div>

            {% endif %}


            {% if answer %}

                <div class="answer">

                    <div class="answer-title">
                        Answer
                    </div>

                    {{ answer }}

                </div>

            {% endif %}


            {% if error_message %}

                <div class="error">

                    {{ error_message }}

                </div>

            {% endif %}


        {% else %}

            <div class="error">

                No document selected.

            </div>

        {% endif %}


        <a
            href="{{ url_for('documents') }}"
            class="back"
        >
            ← Back to Documents
        </a>

    </div>

</div>


</body>

</html>
"""


# ============================================================
# ASKWISE HOME / QUESTION ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    if not login_required():
        return redirect(
            url_for("login")
        )

    answer = None
    error_message = None

    document_id = request.args.get(
        "document_id"
    )

    # --------------------------------------------------------
    # HANDLE QUESTION
    # --------------------------------------------------------

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

        document_id = request.form.get(
            "document_id"
        )

        print()
        print("=" * 70)
        print("NEW QUESTION")
        print("=" * 70)

        print(
            f"Question: {question}"
        )

        print(
            f"Document ID: {document_id}"
        )

        # ----------------------------------------------------
        # VALIDATE QUESTION
        # ----------------------------------------------------

        if not question:

            error_message = (
                "Please enter a question."
            )

        elif not document_id:

            error_message = (
                "No document selected."
            )

        else:

            selected_document = (
                get_document(
                    document_id
                )
            )

            # ------------------------------------------------
            # DOCUMENT EXISTS?
            # ------------------------------------------------

            if not selected_document:

                error_message = (
                    "Document not found."
                )

            # ------------------------------------------------
            # DOCUMENT APPROVED?
            # ------------------------------------------------

            elif (
                selected_document[3]
                != "Approved"
            ):

                error_message = (
                    "This document is not ready "
                    "for questions."
                )

            else:

                try:

                    # ----------------------------------------
                    # INITIALIZE PINECONE
                    # ----------------------------------------

                    store = (
                        initialize_vectorstore()
                    )

                    # ----------------------------------------
                    # DOCUMENT-SPECIFIC RETRIEVER
                    # ----------------------------------------

                    document_retriever = (
                        DocumentRetriever(
                            store,
                            document_id
                        )
                    )

                    # ----------------------------------------
                    # CREATE RAG PIPELINE
                    # ----------------------------------------

                    print()
                    print(
                        "CREATING RAG PIPELINE"
                    )

                    document_rag = (
                        RAGPipeline(
                            document_retriever,
                            GroqLLM()
                        )
                    )

                    # ----------------------------------------
                    # ASK QUESTION
                    # ----------------------------------------

                    print()
                    print(
                        "SENDING QUESTION TO RAG"
                    )

                    response = (
                        document_rag.ask(
                            question
                        )
                    )

                    # ----------------------------------------
                    # HANDLE RESPONSE
                    # ----------------------------------------

                    if isinstance(
                        response,
                        str
                    ):

                        answer = response

                    elif hasattr(
                        response,
                        "content"
                    ):

                        answer = (
                            response.content
                        )

                    else:

                        answer = str(
                            response
                        )

                    print()
                    print(
                        "RAG ANSWER GENERATED"
                    )

                    print(
                        "=" * 70
                    )

                except Exception as e:

                    print()
                    print("=" * 70)
                    print("RAG ERROR")
                    print("=" * 70)

                    print(
                        str(e)
                    )

                    print()
                    print(
                        "FULL TRACEBACK:"
                    )

                    traceback.print_exc()

                    print(
                        "=" * 70
                    )

                    error_message = (
                        "Error while processing "
                        "your question: "
                        + str(e)
                    )

    # --------------------------------------------------------
    # LOAD SELECTED DOCUMENT FOR PAGE
    # --------------------------------------------------------

    selected_document = None

    if document_id:

        selected_document = (
            get_document(
                document_id
            )
        )

    # --------------------------------------------------------
    # RENDER ASKWISE
    # --------------------------------------------------------

    return render_template_string(
        PAGE,
        answer=answer,
        error_message=error_message,
        selected_document=selected_document,
        document_id=document_id
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
    @app.route("/api/document/<int:document_id>", methods=["GET"])
def api_document(document_id):

    if "user_id" not in session:
        return {
            "error": "Unauthorized"
        }, 401

    document = get_document(document_id)

    if not document:
        return {
            "error": "Document not found"
        }, 404

    return {
        "id": document[0],
        "filename": document[1],
        "version": document[2],
        "status": document[3]
    }
    @app.route("/api/health", methods=["GET"])
def api_health():

    return {
        "status": "ok",
        "message": "Askwise Flask API is running"
    }
    @app.route("/api/document/<int:document_id>", methods=["GET"])
def api_document(document_id):

    if "user_id" not in session:
        return {
            "error": "Unauthorized"
        }, 401

    document = get_document(document_id)

    if not document:
        return {
            "error": "Document not found"
        }, 404

    return {
        "id": document[0],
        "filename": document[1],
        "version": document[2],
        "status": document[3]
    }


@app.route("/api/health", methods=["GET"])
def api_health():

    return {
        "status": "ok",
        "message": "Askwise Flask API is running"
    }


if __name__ == "__main__":
    app.run(
        debug=True
    )