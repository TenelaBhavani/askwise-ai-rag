import os
import sqlite3
import traceback
import atexit

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from flask import (
    Flask,
    request,
    session,
    redirect,
    url_for,
    jsonify,
    render_template
)

from flask_cors import CORS

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


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "askwise-development-secret-key"
)

CORS(
    app,
    supports_credentials=True,
    origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174"
    ]
)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "data"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Maximum PDF size = 25 MB
app.config["MAX_CONTENT_LENGTH"] = (
    25 * 1024 * 1024
)


# =========================================================
# DATABASE
# =========================================================

DB_PATH = os.path.join(
    BASE_DIR,
    "users.db"
)


def get_db():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30
    )

    conn.row_factory = sqlite3.Row

    return conn


def init_db():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            version TEXT NOT NULL DEFAULT 'v1.0',
            status TEXT NOT NULL DEFAULT 'Pending',
            chunks INTEGER DEFAULT 0,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


init_db()


# =========================================================
# REQUEST LOGGING
# =========================================================

@app.before_request
def log_request():

    print(
        "\n" + "=" * 60,
        flush=True
    )

    print(
        "INCOMING REQUEST",
        flush=True
    )

    print(
        "Method:",
        request.method,
        flush=True
    )

    print(
        "Path:",
        request.path,
        flush=True
    )

    print(
        "Content Length:",
        request.content_length,
        flush=True
    )

    print(
        "=" * 60,
        flush=True
    )


@app.after_request
def log_response(response):

    print(
        f"RESPONSE: {request.method} "
        f"{request.path} -> "
        f"{response.status_code}",
        flush=True
    )

    return response


# =========================================================
# BACKGROUND EXECUTOR
# =========================================================
#
# IMPORTANT:
#
# We are NOT using:
#
# Thread(..., daemon=True)
#
# anymore.
#
# ThreadPoolExecutor gives us a stable background
# worker for PDF processing while the upload API
# immediately returns to React.
#
# =========================================================

executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="askwise-worker"
)


@atexit.register
def shutdown_executor():

    print(
        "\nShutting down Askwise background workers...",
        flush=True
    )

    executor.shutdown(
        wait=False,
        cancel_futures=True
    )


# =========================================================
# RAG GLOBAL OBJECTS
# =========================================================

embedding = None
vectorstore = None

embedding_lock = Lock()
vectorstore_lock = Lock()


# =========================================================
# DOCUMENT HELPERS
# =========================================================

def get_document(document_id):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            filename,
            version,
            status,
            chunks,
            error,
            created_at
        FROM documents
        WHERE id = ?
    """, (document_id,))

    document = cursor.fetchone()

    conn.close()

    return document


def document_to_dict(document):

    if not document:
        return None

    return {
        "id": document["id"],
        "filename": document["filename"],
        "version": document["version"],
        "status": document["status"],
        "chunks": document["chunks"] or 0,
        "error": document["error"],
        "created_at": document["created_at"]
    }


def update_document_status(
    document_id,
    status,
    chunks=0,
    error=None
):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE documents
        SET
            status = ?,
            chunks = ?,
            error = ?
        WHERE id = ?
    """, (
        status,
        chunks,
        error,
        document_id
    ))

    conn.commit()
    conn.close()

    print(
        f"DATABASE UPDATED -> "
        f"Document {document_id}: "
        f"{status}, chunks={chunks}",
        flush=True
    )


def find_document_file(
    document_id,
    filename
):

    prefixed_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        f"{document_id}_{filename}"
    )

    original_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if os.path.exists(prefixed_path):
        return prefixed_path

    if os.path.exists(original_path):
        return original_path

    return None


# =========================================================
# EMBEDDING MODEL
# =========================================================

def get_embedding():

    global embedding

    if embedding is None:

        with embedding_lock:

            if embedding is None:

                print(
                    "\n" + "=" * 60,
                    flush=True
                )

                print(
                    "LOADING EMBEDDING MODEL",
                    flush=True
                )

                print(
                    "=" * 60,
                    flush=True
                )

                embedding = (
                    SentenceEmbedding(
                        EMBEDDING_MODEL
                    ).get_model()
                )

                print(
                    "Embedding model loaded successfully.",
                    flush=True
                )

    return embedding


# =========================================================
# PINECONE
# =========================================================

def get_vectorstore():

    global vectorstore

    if vectorstore is None:

        with vectorstore_lock:

            if vectorstore is None:

                print(
                    "\n" + "=" * 60,
                    flush=True
                )

                print(
                    "CONNECTING TO PINECONE",
                    flush=True
                )

                print(
                    "=" * 60,
                    flush=True
                )

                embedding_model = (
                    get_embedding()
                )

                store = PineconeStore(
                    PINECONE_INDEX,
                    embedding_model
                )

                vectorstore = (
                    store.connect()
                )

                print(
                    "Pinecone connected successfully.",
                    flush=True
                )

    return vectorstore


# =========================================================
# PDF INGESTION
# =========================================================

def ingest_pdf(
    pdf_path,
    document_id,
    filename
):

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "STARTING DOCUMENT INGESTION",
        flush=True
    )

    print(
        "Document ID:",
        document_id,
        flush=True
    )

    print(
        "Filename:",
        filename,
        flush=True
    )

    print(
        "PDF Path:",
        pdf_path,
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    # -----------------------------------------------------
    # STEP 1 - LOAD PDF
    # -----------------------------------------------------

    print(
        "\n[1/5] Loading PDF...",
        flush=True
    )

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            f"PDF file does not exist: {pdf_path}"
        )

    file_size = os.path.getsize(
        pdf_path
    )

    print(
        f"PDF size: {file_size} bytes",
        flush=True
    )

    if file_size == 0:

        raise ValueError(
            "Uploaded PDF is empty."
        )

    pdf_loader = PDFLoader()

    loaded_documents = (
        pdf_loader.load(
            pdf_path
        )
    )

    print(
        f"Loaded "
        f"{len(loaded_documents)} "
        f"document section(s).",
        flush=True
    )

    if not loaded_documents:

        raise ValueError(
            "No readable content found in the PDF."
        )

    # -----------------------------------------------------
    # STEP 2 - SPLIT INTO CHUNKS
    # -----------------------------------------------------

    print(
        "\n[2/5] Splitting PDF into chunks...",
        flush=True
    )

    text_splitter = TextSplitter(
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    documents = (
        text_splitter.split(
            loaded_documents
        )
    )

    print(
        f"Created "
        f"{len(documents)} chunks.",
        flush=True
    )

    if not documents:

        raise ValueError(
            "No text chunks were generated from the PDF."
        )

    # -----------------------------------------------------
    # STEP 3 - ADD METADATA
    # -----------------------------------------------------

    print(
        "\n[3/5] Adding metadata...",
        flush=True
    )

    for index, document in enumerate(
        documents
    ):

        if (
            not hasattr(
                document,
                "metadata"
            )
            or document.metadata is None
        ):

            document.metadata = {}

        document.metadata[
            "document_id"
        ] = str(
            document_id
        )

        document.metadata[
            "filename"
        ] = filename

        document.metadata[
            "chunk_id"
        ] = index

    print(
        "Metadata added successfully.",
        flush=True
    )

    # -----------------------------------------------------
    # STEP 4 - EMBEDDING MODEL
    # -----------------------------------------------------

    print(
        "\n[4/5] Creating embeddings...",
        flush=True
    )

    embedding_model = (
        get_embedding()
    )

    print(
        "Embedding model ready.",
        flush=True
    )

    # -----------------------------------------------------
    # STEP 5 - STORE IN PINECONE
    # -----------------------------------------------------

    print(
        "\n[5/5] Storing chunks in Pinecone...",
        flush=True
    )

    store = PineconeStore(
        PINECONE_INDEX,
        embedding_model
    )

    store.create_store(
        documents
    )

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "DOCUMENT INGESTION SUCCESSFUL",
        flush=True
    )

    print(
        "Document:",
        filename,
        flush=True
    )

    print(
        "Chunks:",
        len(documents),
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    return len(documents)
    # =========================================================
# BACKGROUND PDF PROCESSING
# =========================================================

def process_document_background(
    pdf_path,
    document_id,
    filename
):

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "BACKGROUND PROCESSING STARTED",
        flush=True
    )

    print(
        "Document ID:",
        document_id,
        flush=True
    )

    print(
        "Filename:",
        filename,
        flush=True
    )

    print(
        "PDF Path:",
        pdf_path,
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    try:

        # -------------------------------------------------
        # VERIFY FILE
        # -------------------------------------------------

        print(
            "\n[BACKGROUND 0] "
            "Checking PDF file...",
            flush=True
        )

        if not os.path.exists(
            pdf_path
        ):

            raise FileNotFoundError(
                f"PDF file not found: "
                f"{pdf_path}"
            )

        print(
            "PDF file exists.",
            flush=True
        )

        # -------------------------------------------------
        # INGEST
        # -------------------------------------------------

        print(
            "\n[BACKGROUND 1] "
            "Starting PDF ingestion...",
            flush=True
        )

        chunks_count = ingest_pdf(
            pdf_path,
            document_id,
            filename
        )

        # -------------------------------------------------
        # VERIFY CHUNKS
        # -------------------------------------------------

        if chunks_count <= 0:

            raise ValueError(
                "No chunks were generated."
            )

        print(
            "\n[BACKGROUND 2] "
            "PDF ingestion completed.",
            flush=True
        )

        print(
            "Chunks generated:",
            chunks_count,
            flush=True
        )

        # -------------------------------------------------
        # UPDATE DATABASE
        # -------------------------------------------------

        update_document_status(
            document_id,
            "Approved",
            chunks_count,
            None
        )

        print(
            "\n[BACKGROUND 3] "
            "Document status updated "
            "to APPROVED.",
            flush=True
        )

        print(
            "Document ID:",
            document_id,
            flush=True
        )

        print(
            "Chunks:",
            chunks_count,
            flush=True
        )

        print(
            "\nASK BUTTON IS NOW AVAILABLE",
            flush=True
        )

    except Exception as e:

        error_message = str(
            e
        )

        print(
            "\n" + "=" * 70,
            flush=True
        )

        print(
            "BACKGROUND PROCESSING FAILED",
            flush=True
        )

        print(
            "Document ID:",
            document_id,
            flush=True
        )

        print(
            "Error:",
            error_message,
            flush=True
        )

        print(
            "=" * 70,
            flush=True
        )

        traceback.print_exc()

        try:

            update_document_status(
                document_id,
                "Failed",
                0,
                error_message
            )

        except Exception:

            print(
                "Could not update "
                "document status to Failed.",
                flush=True
            )

            traceback.print_exc()

    finally:

        print(
            "\nBACKGROUND PROCESSING FINISHED:",
            document_id,
            flush=True
        )


# =========================================================
# SUBMIT BACKGROUND JOB
# =========================================================

def submit_document_processing(
    pdf_path,
    document_id,
    filename
):

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "SUBMITTING BACKGROUND JOB",
        flush=True
    )

    print(
        "Document ID:",
        document_id,
        flush=True
    )

    print(
        "Filename:",
        filename,
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    try:

        future = executor.submit(
            process_document_background,
            pdf_path,
            document_id,
            filename
        )

        print(
            "BACKGROUND JOB SUBMITTED SUCCESSFULLY",
            flush=True
        )

        print(
            "Future:",
            future,
            flush=True
        )

        return future

    except Exception as e:

        print(
            "\nFAILED TO SUBMIT "
            "BACKGROUND JOB",
            flush=True
        )

        traceback.print_exc()

        raise e


# =========================================================
# HEALTH
# =========================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def api_health():

    return jsonify({
        "status": "ok",
        "message": (
            "Askwise Flask API is running"
        )
    })


# =========================================================
# REGISTER API
# =========================================================

@app.route(
    "/api/register",
    methods=["POST"]
)
def api_register():

    data = (
        request.get_json(
            silent=True
        ) or {}
    )

    username = data.get(
        "username",
        ""
    ).strip()

    password = data.get(
        "password",
        ""
    )

    if (
        not username
        or not password
    ):

        return jsonify({
            "success": False,
            "error": (
                "Username and password "
                "are required."
            )
        }), 400

    if len(password) < 6:

        return jsonify({
            "success": False,
            "error": (
                "Password must contain "
                "at least 6 characters."
            )
        }), 400

    hashed_password = (
        generate_password_hash(
            password
        )
    )

    try:

        conn = get_db()
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

        user_id = (
            cursor.lastrowid
        )

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": (
                "Registration successful."
            ),
            "user": {
                "id": user_id,
                "username": username
            }
        }), 201

    except sqlite3.IntegrityError:

        return jsonify({
            "success": False,
            "error": (
                "Username already exists."
            )
        }), 409


# =========================================================
# LOGIN API
# =========================================================

@app.route(
    "/api/login",
    methods=["POST"]
)
def api_login():

    data = (
        request.get_json(
            silent=True
        ) or {}
    )

    username = data.get(
        "username",
        ""
    ).strip()

    password = data.get(
        "password",
        ""
    )

    if (
        not username
        or not password
    ):

        return jsonify({
            "success": False,
            "error": (
                "Username and password "
                "are required."
            )
        }), 400

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            username,
            password
        FROM users
        WHERE username = ?
    """, (
        username,
    ))

    user = cursor.fetchone()

    conn.close()

    if (
        not user
        or not check_password_hash(
            user["password"],
            password
        )
    ):

        return jsonify({
            "success": False,
            "error": (
                "Invalid username or password."
            )
        }), 401

    session.clear()

    session["user_id"] = (
        user["id"]
    )

    session["username"] = (
        user["username"]
    )

    return jsonify({
        "success": True,
        "message": (
            "Login successful."
        ),
        "user": {
            "id": user["id"],
            "username": user["username"]
        }
    })


# =========================================================
# CURRENT USER
# =========================================================

@app.route(
    "/api/me",
    methods=["GET"]
)
def api_me():

    if (
        "user_id"
        not in session
    ):

        return jsonify({
            "success": False,
            "authenticated": False
        }), 401

    return jsonify({
        "success": True,
        "authenticated": True,
        "user": {
            "id": session[
                "user_id"
            ],
            "username": session[
                "username"
            ]
        }
    })


# =========================================================
# LOGOUT API
# =========================================================

@app.route(
    "/api/logout",
    methods=["POST"]
)
def api_logout():

    session.clear()

    return jsonify({
        "success": True,
        "message": (
            "Logout successful."
        )
    })


# =========================================================
# REACT UPLOAD API
# =========================================================

@app.route(
    "/api/upload-document",
    methods=["POST"]
)
def api_upload_document():

    document_id = None
    filepath = None

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "UPLOAD API REQUEST RECEIVED",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    try:

        # -------------------------------------------------
        # STEP 1 - AUTHENTICATION
        # -------------------------------------------------

        if (
            "user_id"
            not in session
        ):

            print(
                "ERROR: User not authenticated",
                flush=True
            )

            return jsonify({
                "success": False,
                "error": (
                    "Please login first."
                )
            }), 401

        print(
            "STEP 1: User authenticated",
            flush=True
        )

        # -------------------------------------------------
        # STEP 2 - GET PDF
        # -------------------------------------------------

        file = request.files.get(
            "document"
        )

        if (
            not file
            or not file.filename
        ):

            return jsonify({
                "success": False,
                "error": (
                    "Please select a PDF document."
                )
            }), 400

        filename = secure_filename(
            file.filename
        )

        if not filename:

            return jsonify({
                "success": False,
                "error": (
                    "Invalid filename."
                )
            }), 400

        extension = (
            os.path.splitext(
                filename
            )[1].lower()
        )

        if extension != ".pdf":

            return jsonify({
                "success": False,
                "error": (
                    "Only PDF files are supported."
                )
            }), 400

        print(
            "STEP 2: PDF received:",
            filename,
            flush=True
        )

        # -------------------------------------------------
        # STEP 3 - DATABASE RECORD
        # -------------------------------------------------

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO documents (
                filename,
                version,
                status,
                chunks,
                error
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            filename,
            "v1.0",
            "Processing",
            0,
            None
        ))

        document_id = (
            cursor.lastrowid
        )

        conn.commit()
        conn.close()

        print(
            "STEP 3: Database record created",
            flush=True
        )

        print(
            "Document ID:",
            document_id,
            flush=True
        )

        # -------------------------------------------------
        # STEP 4 - SAVE PDF
        # -------------------------------------------------

        stored_filename = (
            f"{document_id}_{filename}"
        )

        filepath = os.path.join(
            app.config[
                "UPLOAD_FOLDER"
            ],
            stored_filename
        )

        file.save(
            filepath
        )

        if not os.path.exists(
            filepath
        ):

            raise FileNotFoundError(
                "PDF was not saved successfully."
            )

        file_size = os.path.getsize(
            filepath
        )

        if file_size <= 0:

            raise ValueError(
                "Uploaded PDF is empty."
            )

        print(
            "STEP 4: PDF saved successfully",
            flush=True
        )

        print(
            "Saved path:",
            filepath,
            flush=True
        )

        print(
            "File size:",
            file_size,
            "bytes",
            flush=True
        )

        # -------------------------------------------------
        # STEP 5 - BACKGROUND PROCESSING
        # -------------------------------------------------

        print(
            "\nSTEP 5: Submitting "
            "background worker...",
            flush=True
        )

        submit_document_processing(
            filepath,
            document_id,
            filename
        )

        print(
            "STEP 5: Background worker "
            "submitted successfully",
            flush=True
        )

        # -------------------------------------------------
        # IMPORTANT
        # -------------------------------------------------
        #
        # DO NOT WAIT HERE.
        #
        # React gets the 202 response immediately.
        #
        # Background worker continues:
        #
        # Processing
        #      ↓
        # Load PDF
        #      ↓
        # Split chunks
        #      ↓
        # Embeddings
        #      ↓
        # Pinecone
        #      ↓
        # Approved
        #
        # -------------------------------------------------

        return jsonify({
            "success": True,
            "message": (
                "PDF uploaded. "
                "Processing has started."
            ),
            "document": {
                "id": document_id,
                "filename": filename,
                "version": "v1.0",
                "status": "Processing",
                "chunks": 0
            }
        }), 202

    except Exception as e:

        print(
            "\n" + "=" * 70,
            flush=True
        )

        print(
            "UPLOAD PROCESSING ERROR",
            flush=True
        )

        print(
            "Error:",
            str(e),
            flush=True
        )

        print(
            "=" * 70,
            flush=True
        )

        traceback.print_exc()

        if document_id is not None:

            try:

                update_document_status(
                    document_id,
                    "Failed",
                    0,
                    str(e)
                )

            except Exception:

                traceback.print_exc()

        return jsonify({
            "success": False,
            "message": (
                "Document upload failed."
            ),
            "error": str(e)
        }), 500


# =========================================================
# DOCUMENT LIST API
# =========================================================

@app.route(
    "/api/documents",
    methods=["GET"]
)
def api_documents():

    if (
        "user_id"
        not in session
    ):

        return jsonify({
            "success": False,
            "error": (
                "Please login first."
            )
        }), 401

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            filename,
            version,
            status,
            chunks,
            error,
            created_at
        FROM documents
        ORDER BY id DESC
    """)

    documents = (
        cursor.fetchall()
    )

    conn.close()

    return jsonify({
        "success": True,
        "documents": [
            document_to_dict(
                document
            )
            for document in documents
        ]
    })


# =========================================================
# DOCUMENT DETAIL API
# =========================================================

@app.route(
    "/api/document/<int:document_id>",
    methods=["GET"]
)
def api_document(
    document_id
):

    document = get_document(
        document_id
    )

    if not document:

        return jsonify({
            "success": False,
            "error": (
                "Document not found."
            )
        }), 404

    return jsonify({
        "success": True,
        "document": (
            document_to_dict(
                document
            )
        )
    })


# =========================================================
# DELETE DOCUMENT API
# =========================================================

@app.route(
    "/api/document/<int:document_id>",
    methods=["DELETE"]
)
def api_delete_document(
    document_id
):

    if (
        "user_id"
        not in session
    ):

        return jsonify({
            "success": False,
            "error": (
                "Please login first."
            )
        }), 401

    document = get_document(
        document_id
    )

    if not document:

        return jsonify({
            "success": False,
            "error": (
                "Document not found."
            )
        }), 404

    filename = (
        document["filename"]
    )

    filepath = (
        find_document_file(
            document_id,
            filename
        )
    )

    try:

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM documents
            WHERE id = ?
        """, (
            document_id,
        ))

        conn.commit()
        conn.close()

        if (
            filepath
            and os.path.exists(
                filepath
            )
        ):

            os.remove(
                filepath
            )

        print(
            "Document deleted:",
            document_id,
            flush=True
        )

        return jsonify({
            "success": True,
            "message": (
                "Document deleted successfully."
            )
        })

    except Exception as e:

        print(
            "DELETE DOCUMENT ERROR",
            flush=True
        )

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
        # =========================================================
# ASK API
# =========================================================

@app.route(
    "/api/ask",
    methods=["POST"]
)
def api_ask():

    if (
        "user_id"
        not in session
    ):

        return jsonify({
            "success": False,
            "error": (
                "Please login first."
            )
        }), 401

    data = (
        request.get_json(
            silent=True
        ) or {}
    )

    document_id = (
        data.get(
            "document_id"
        )
    )

    question = (
        data.get(
            "question",
            ""
        )
        .strip()
    )

    if not document_id:

        return jsonify({
            "success": False,
            "error": (
                "Document ID is required."
            )
        }), 400

    if not question:

        return jsonify({
            "success": False,
            "error": (
                "Question is required."
            )
        }), 400

    document = get_document(
        document_id
    )

    if not document:

        return jsonify({
            "success": False,
            "error": (
                "Document not found."
            )
        }), 404

    if document["status"] != "Approved":

        return jsonify({
            "success": False,
            "error": (
                "This document is not "
                "ready for questions."
            ),
            "status": document[
                "status"
            ]
        }), 400

    try:

        print(
            "\n" + "=" * 60,
            flush=True
        )

        print(
            "RAG QUESTION",
            flush=True
        )

        print(
            "Document ID:",
            document_id,
            flush=True
        )

        print(
            "Question:",
            question,
            flush=True
        )

        print(
            "=" * 60,
            flush=True
        )

        # -------------------------------------------------
        # GET VECTOR STORE
        # -------------------------------------------------

        store = (
            get_vectorstore()
        )

        # -------------------------------------------------
        # RETRIEVER
        # -------------------------------------------------

        retriever = (
            store.as_retriever(
                search_kwargs={
                    "k": TOP_K,
                    "filter": {
                        "document_id": {
                            "$eq": str(
                                document_id
                            )
                        }
                    }
                }
            )
        )

        print(
            "Retrieving relevant chunks...",
            flush=True
        )

        retrieved_documents = (
            retriever.invoke(
                question
            )
        )

        retrieval_scores = []

        try:
            scored_documents = (
                store.similarity_search_with_score(
                    question,
                    k=TOP_K,
                    filter={
                        "document_id": {
                            "$eq": str(
                                document_id
                            )
                        }
                    }
                )
            )

            scored_documents = [
                scored_document
                for scored_document in scored_documents
                if str(
                    (
                        getattr(
                            scored_document[0],
                            "metadata",
                            {}
                        ) or {}
                    ).get(
                        "document_id"
                    )
                ) == str(
                    document_id
                )
            ]

            retrieved_documents = [
                scored_document[0]
                for scored_document in scored_documents
            ]
            retrieval_scores = [
                float(
                    scored_document[1]
                )
                for scored_document in scored_documents
            ]

        except Exception as score_error:

            print(
                "Similarity scores unavailable:",
                str(score_error),
                flush=True
            )

        retrieved_documents = [
            retrieved_document
            for retrieved_document in retrieved_documents
            if str(
                (
                    getattr(
                        retrieved_document,
                        "metadata",
                        {}
                    ) or {}
                ).get(
                    "document_id"
                )
            ) == str(
                document_id
            )
        ]

        print(
            "Retrieved chunks:",
            len(
                retrieved_documents
            ),
            flush=True
        )

        # -------------------------------------------------
        # SOURCES
        # -------------------------------------------------

        sources = []

        for index, doc in enumerate(
            retrieved_documents
        ):

            metadata = (
                getattr(
                    doc,
                    "metadata",
                    {}
                ) or {}
            )

            page_content = (
                getattr(
                    doc,
                    "page_content",
                    ""
                )
            )

            similarity = (
                retrieval_scores[index]
                if index < len(retrieval_scores)
                else None
            )

            if similarity is not None:
                if similarity < 0:
                    similarity = (
                        similarity + 1
                    ) / 2

                similarity = min(
                    1.0,
                    max(
                        0.0,
                        similarity
                    )
                )

            sources.append({
                "chunk": (
                    index + 1
                ),
                "content": (
                    page_content
                ),
                "metadata": (
                    metadata
                )
            })

        # -------------------------------------------------
        # SIMPLE RETRIEVER
        # -------------------------------------------------

        class SimpleRetriever:

            def __init__(
                self,
                documents
            ):

                self.documents = (
                    documents
                )

            def retrieve(
                self,
                question
            ):

                return (
                    self.documents
                )

        simple_retriever = (
            SimpleRetriever(
                retrieved_documents
            )
        )

        # -------------------------------------------------
        # RAG PIPELINE
        # -------------------------------------------------

        rag = RAGPipeline(
            simple_retriever,
            GroqLLM()
        )

        print(
            "Generating answer...",
            flush=True
        )

        response = (
            rag.ask(
                question
            )
        )

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

        print(
            "Answer generated successfully.",
            flush=True
        )

        return jsonify({
            "success": True,
            "answer": answer,
            "document": (
                document_to_dict(
                    document
                )
            ),
            "sources": sources,
            "chunks_used": len(
                retrieved_documents
            )
        })

    except Exception as e:

        print(
            "\n" + "=" * 60,
            flush=True
        )

        print(
            "RAG ERROR",
            flush=True
        )

        print(
            "Error:",
            str(e),
            flush=True
        )

        print(
            "=" * 60,
            flush=True
        )

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# OLD HTML LOGIN
# =========================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = (
            request.form.get(
                "username",
                ""
            ).strip()
        )

        password = (
            request.form.get(
                "password",
                ""
            )
        )

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                id,
                username,
                password
            FROM users
            WHERE username = ?
        """, (
            username,
        ))

        user = (
            cursor.fetchone()
        )

        conn.close()

        if (
            user
            and check_password_hash(
                user["password"],
                password
            )
        ):

            session["user_id"] = (
                user["id"]
            )

            session["username"] = (
                user["username"]
            )

            return redirect(
                url_for(
                    "documents"
                )
            )

        return (
            "Invalid username "
            "or password."
        )

    return render_template(
        "login.html"
    )


# =========================================================
# OLD HTML REGISTER
# =========================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        username = (
            request.form.get(
                "username",
                ""
            ).strip()
        )

        password = (
            request.form.get(
                "password",
                ""
            )
        )

        if (
            not username
            or not password
        ):

            return (
                "Username and "
                "password are required."
            )

        hashed_password = (
            generate_password_hash(
                password
            )
        )

        try:

            conn = get_db()
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

            return redirect(
                url_for(
                    "login"
                )
            )

        except sqlite3.IntegrityError:

            return (
                "Username already exists."
            )

    return render_template(
        "register.html"
    )


# =========================================================
# OLD LOGOUT
# =========================================================

@app.route(
    "/logout"
)
def logout():

    session.clear()

    return redirect(
        url_for(
            "login"
        )
    )


# =========================================================
# OLD DOCUMENTS PAGE
# =========================================================

@app.route(
    "/documents"
)
def documents():

    if (
        "user_id"
        not in session
    ):

        return redirect(
            url_for(
                "login"
            )
        )

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM documents
        ORDER BY id DESC
    """)

    documents = (
        cursor.fetchall()
    )

    conn.close()

    return render_template(
        "documents.html",
        documents=documents
    )


# =========================================================
# OLD HTML UPLOAD
# =========================================================

@app.route(
    "/upload-document",
    methods=["POST"]
)
def upload_document():

    if (
        "user_id"
        not in session
    ):

        return redirect(
            url_for(
                "login"
            )
        )

    file = request.files.get(
        "document"
    )

    if (
        not file
        or not file.filename
    ):

        return (
            "Please select a PDF."
        )

    filename = secure_filename(
        file.filename
    )

    if not filename:

        return (
            "Invalid filename."
        )

    if (
        os.path.splitext(
            filename
        )[1].lower()
        != ".pdf"
    ):

        return (
            "Only PDF files "
            "are supported."
        )

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO documents (
            filename,
            version,
            status,
            chunks,
            error
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        filename,
        "v1.0",
        "Processing",
        0,
        None
    ))

    document_id = (
        cursor.lastrowid
    )

    conn.commit()
    conn.close()

    stored_filename = (
        f"{document_id}_{filename}"
    )

    filepath = os.path.join(
        app.config[
            "UPLOAD_FOLDER"
        ],
        stored_filename
    )

    try:

        file.save(
            filepath
        )

        chunks_count = (
            ingest_pdf(
                filepath,
                document_id,
                filename
            )
        )

        update_document_status(
            document_id,
            "Approved",
            chunks_count,
            None
        )

        return redirect(
            url_for(
                "documents"
            )
        )

    except Exception as e:

        traceback.print_exc()

        update_document_status(
            document_id,
            "Failed",
            0,
            str(e)
        )

        return (
            "Document processing failed: "
            + str(e)
        )


# =========================================================
# OLD HTML DELETE
# =========================================================

@app.route(
    "/delete-document/<int:document_id>",
    methods=["POST"]
)
def delete_document(
    document_id
):

    if (
        "user_id"
        not in session
    ):

        return redirect(
            url_for(
                "login"
            )
        )

    document = get_document(
        document_id
    )

    if document:

        filename = (
            document["filename"]
        )

        filepath = (
            find_document_file(
                document_id,
                filename
            )
        )

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM documents
            WHERE id = ?
        """, (
            document_id,
        ))

        conn.commit()
        conn.close()

        if (
            filepath
            and os.path.exists(
                filepath
            )
        ):

            try:

                os.remove(
                    filepath
                )

            except Exception:

                traceback.print_exc()

    return redirect(
        url_for(
            "documents"
        )
    )


# =========================================================
# ROOT
# =========================================================

@app.route("/")
def home():

    return jsonify({
        "status": "ok",
        "message": (
            "Askwise Flask API is running"
        )
    })


# =========================================================
# 413 - FILE TOO LARGE
# =========================================================

@app.errorhandler(413)
def file_too_large(
    error
):

    return jsonify({
        "success": False,
        "error": (
            "PDF is too large. "
            "Maximum allowed size "
            "is 25 MB."
        )
    }), 413


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "ASKWISE FLASK SERVER STARTING",
        flush=True
    )

    print(
        "Host: 0.0.0.0",
        flush=True
    )

    print(
        "Port: 5001",
        flush=True
    )

    print(
        "Background workers: 2",
        flush=True
    )

    print(
        "Debug: False",
        flush=True
    )

    print(
        "Auto reloader: Disabled",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,
        use_reloader=False,
        threaded=True
    )
    