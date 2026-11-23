
import { useEffect, useRef, useState } from "react";

/* =========================================================
   API
========================================================= */

const API = "http://localhost:5001";


/* =========================================================
   NAVIGATION
========================================================= */

function navigate(path) {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}


function ThemeToggle() {
  const [theme, setTheme] = useState(() => {
    const savedTheme = window.localStorage.getItem("askwise-theme");
    return savedTheme === "light" || savedTheme === "dark"
      ? savedTheme
      : "dark";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    window.localStorage.setItem("askwise-theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((currentTheme) => (
      currentTheme === "dark" ? "light" : "dark"
    ));
  };

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      className="theme-toggle"
    >
      {theme === "dark" ? (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M20.7 15.1A8.5 8.5 0 0 1 8.9 3.3a8.5 8.5 0 1 0 11.8 11.8Z" />
        </svg>
      )}
    </button>
  );
}


/* =========================================================
   LOGIN PAGE
========================================================= */

function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const login = async (e) => {
    e.preventDefault();

    setError("");
    setLoading(true);

    try {
      const response = await fetch(`${API}/api/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({
          username,
          password,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        setError(data.error || "Invalid username or password.");
        return;
      }

      navigate("/upload");

    } catch (error) {
      console.error(error);
      setError("Unable to connect to Askwise server.");

    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#050816] text-white relative overflow-hidden">
      <div className="fixed top-5 right-5 z-50">
        <ThemeToggle />
      </div>

      <div className="absolute top-[-200px] left-[-200px] w-[500px] h-[500px] bg-blue-600/20 rounded-full blur-[120px]" />

      <div className="absolute bottom-[-200px] right-[-200px] w-[500px] h-[500px] bg-purple-600/20 rounded-full blur-[120px]" />

      <div className="relative min-h-screen flex items-center justify-center px-6">

        <div className="w-full max-w-5xl grid md:grid-cols-2 gap-12 items-center">

          <div className="hidden md:block">

            <div className="mb-8">

              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-sm text-blue-300 mb-6">
                <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                AI Document Intelligence
              </div>

              <h1 className="text-6xl font-bold leading-tight">
                Ask
                <span className="text-blue-400">wise</span>
              </h1>

              <p className="text-slate-400 text-lg mt-6 max-w-lg leading-relaxed">
                Upload your documents, understand them
                with AI, and get intelligent answers
                instantly.
              </p>

            </div>

            <div className="grid grid-cols-3 gap-4">

              <Feature
                icon="📄"
                title="Upload"
                text="PDF documents"
              />

              <Feature
                icon="🧠"
                title="AI"
                text="RAG powered"
              />

              <Feature
                icon="💬"
                title="Ask"
                text="Natural answers"
              />

            </div>

          </div>


          <div className="bg-white/[0.06] backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl">

            <div className="mb-8">

              <div className="text-3xl font-bold">
                Welcome back
              </div>

              <p className="text-slate-400 mt-2">
                Sign in to continue to Askwise
              </p>

            </div>

            <form onSubmit={login} className="space-y-5">

              <div>
                <label className="text-sm text-slate-300">
                  Username
                </label>

                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="mt-2 w-full bg-black/30 border border-white/10 rounded-xl px-4 py-3 outline-none focus:border-blue-500 transition"
                  placeholder="Enter your username"
                  required
                />
              </div>

              <div>
                <label className="text-sm text-slate-300">
                  Password
                </label>

                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="mt-2 w-full bg-black/30 border border-white/10 rounded-xl px-4 py-3 outline-none focus:border-blue-500 transition"
                  placeholder="Enter your password"
                  required
                />
              </div>

              {error && (
                <div className="bg-red-500/10 border border-red-500/20 text-red-300 rounded-xl px-4 py-3 text-sm">
                  {error}
                </div>
              )}

              <button
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-xl py-3.5 font-semibold transition shadow-lg shadow-blue-600/20"
              >
                {loading ? "Signing in..." : "Sign in to Askwise"}
              </button>

            </form>

          </div>

        </div>

      </div>

    </div>
  );
}


/* =========================================================
   FEATURE
========================================================= */

function Feature({ icon, title, text }) {
  return (
    <div className="bg-white/5 border border-white/10 rounded-2xl p-4">

      <div className="text-2xl mb-2">
        {icon}
      </div>

      <div className="font-semibold">
        {title}
      </div>

      <div className="text-xs text-slate-500 mt-1">
        {text}
      </div>

    </div>
  );
}


/* =========================================================
   UPLOAD PAGE
========================================================= */

function UploadPage() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const [documents, setDocuments] = useState([]);
  const [deletingId, setDeletingId] = useState(null);
  const askRefreshTimer = useRef(null);


  /* LOAD DOCUMENT HISTORY */

  const loadDocuments = async () => {
    try {
      const response = await fetch(`${API}/api/documents`, {
        credentials: "include",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Unable to load documents.");
      }

      if (data.success) {
        setDocuments(data.documents || []);
      }

    } catch (error) {
      console.error("LOAD DOCUMENTS ERROR:", error);
    }
  };


  useEffect(() => {
    loadDocuments();

    const refreshInterval = window.setInterval(() => {
      loadDocuments();
    }, 3000);

    return () => {
      window.clearInterval(refreshInterval);
      if (askRefreshTimer.current) {
        window.clearTimeout(askRefreshTimer.current);
      }
    };
  }, []);


  /* UPLOAD DOCUMENT */

  const uploadDocument = async () => {
    if (!file) {
      setMessage("Please select a PDF first.");
      return;
    }

    setUploading(true);
    setMessage("");

    const formData = new FormData();
    formData.append("document", file);

    try {
      const response = await fetch(`${API}/api/upload-document`, {
        method: "POST",
        credentials: "include",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        setMessage(data.error || "Upload failed.");
        await loadDocuments();
        return;
      }

      setMessage("Document uploaded successfully.");
      setFile(null);

      const input = document.getElementById("pdfInput");

      if (input) {
        input.value = "";
      }

      await loadDocuments();
      if (askRefreshTimer.current) {
        window.clearTimeout(askRefreshTimer.current);
      }
      askRefreshTimer.current = window.setTimeout(() => {
        loadDocuments();
      }, 3000);

    } catch (error) {
      console.error("UPLOAD ERROR:", error);
      setMessage("Unable to connect to Flask.");

    } finally {
      setUploading(false);
    }
  };


  /* DELETE DOCUMENT */

  const deleteDocument = async (documentId, filename) => {
    const confirmed = window.confirm(
      `Are you sure you want to delete "${filename}"?`
    );

    if (!confirmed) {
      return;
    }

    setDeletingId(documentId);
    setMessage("");

    try {
      const response = await fetch(
        `${API}/api/document/${documentId}`,
        {
          method: "DELETE",
          credentials: "include",
        }
      );

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        setMessage(data.error || "Unable to delete document.");
        return;
      }

      setMessage("Document deleted successfully.");

      await loadDocuments();

    } catch (error) {
      console.error("DELETE ERROR:", error);
      setMessage("Unable to connect to Flask.");

    } finally {
      setDeletingId(null);
    }
  };


  /* LOGOUT */

  const logout = async () => {
    try {
      await fetch(`${API}/api/logout`, {
        method: "POST",
        credentials: "include",
      });
    } catch (error) {
      console.error("LOGOUT ERROR:", error);
    }

    navigate("/");
  };


  return (
    <div className="min-h-screen bg-[#050816] text-white">

      {/* HEADER */}

      <header className="border-b border-white/10 bg-black/20 backdrop-blur-xl">

        <div className="max-w-7xl mx-auto px-6 py-5 flex items-center justify-between">

          <div>
            <div className="text-2xl font-bold">
              Ask<span className="text-blue-400">wise</span>
            </div>

            <div className="text-xs text-slate-500">
              AI Document Intelligence
            </div>
          </div>

          <div className="flex items-center gap-3">
            <ThemeToggle />
            <button
              onClick={logout}
              className="px-4 py-2 rounded-lg border border-white/10 text-slate-300 hover:bg-white/5 transition"
            >
              Logout
            </button>
          </div>

        </div>

      </header>


      <main className="max-w-7xl mx-auto px-6 py-10">

        {/* HERO */}

        <div className="mb-10">

          <div className="text-blue-400 text-sm font-medium mb-3">
            DOCUMENT WORKSPACE
          </div>

          <h1 className="text-4xl font-bold">
            Upload and understand your documents
          </h1>

          <p className="text-slate-400 mt-3">
            Upload a PDF and use AI to ask questions
            directly from its content.
          </p>

        </div>


        {/* UPLOAD CARD */}

        <div className="bg-white/[0.04] border border-white/10 rounded-3xl p-8">

          <div className="flex flex-col md:flex-row gap-6 items-center">

            <div className="flex-1">

              <div className="text-xl font-semibold mb-2">
                Upload PDF
              </div>

              <p className="text-slate-500 text-sm mb-5">
                Supported format: PDF
              </p>

              <label
                htmlFor="pdfInput"
                className="block border-2 border-dashed border-white/10 hover:border-blue-500/50 rounded-2xl p-8 text-center cursor-pointer transition"
              >

                <div className="text-4xl mb-3">
                  📄
                </div>

                <div className="font-medium">
                  {file ? file.name : "Choose a PDF document"}
                </div>

                <div className="text-sm text-slate-500 mt-2">
                  Click to browse files
                </div>

              </label>

              <input
                id="pdfInput"
                type="file"
                accept=".pdf,application/pdf"
                className="hidden"
                onChange={(e) => setFile(e.target.files[0] || null)}
              />

            </div>


            <div className="w-full md:w-64">

              <button
                onClick={uploadDocument}
                disabled={uploading}
                className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-xl py-4 font-semibold transition shadow-lg shadow-blue-600/20"
              >
                {uploading ? "Processing document..." : "Upload & Process"}
              </button>

              {message && (
                <div className="mt-4 text-sm text-center text-slate-400">
                  {message}
                </div>
              )}

            </div>

          </div>

        </div>


        {/* DOCUMENT HISTORY */}

        <div className="mt-12">

          <div className="flex items-center justify-between mb-5">

            <div>
              <h2 className="text-2xl font-bold">
                Your Documents
              </h2>

              <p className="text-slate-500 text-sm mt-1">
                {documents.length} document(s)
              </p>
            </div>

            <button
              onClick={loadDocuments}
              className="px-4 py-2 rounded-lg border border-white/10 text-slate-300 hover:bg-white/5 transition text-sm"
            >
              ↻ Refresh
            </button>

          </div>


          {documents.length === 0 ? (

            <div className="border border-white/10 rounded-2xl p-12 text-center text-slate-500">
              No documents uploaded yet.
            </div>

          ) : (

            <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-5">

              {documents.map((doc) => (

                <DocumentCard
                  key={doc.id}
                  doc={doc}
                  onDelete={deleteDocument}
                  deleting={deletingId === doc.id}
                />

              ))}

            </div>

          )}

        </div>

      </main>

    </div>
  );
}


/* =========================================================
   DOCUMENT CARD
========================================================= */

function DocumentCard({ doc, onDelete, deleting }) {
  const approved = doc.status === "Approved";

  return (
    <div className="bg-white/[0.04] border border-white/10 hover:border-blue-500/40 rounded-2xl p-6 transition">

      {/* TOP SECTION */}

      <div className="flex items-start justify-between gap-4">

        <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center text-2xl">
          📄
        </div>

        <span
          className={`px-3 py-1 rounded-full text-xs font-medium ${
            approved
              ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
              : doc.status === "Failed"
              ? "bg-red-500/10 text-red-400 border border-red-500/20"
              : "bg-yellow-500/10 text-yellow-400 border border-yellow-500/20"
          }`}
        >
          {doc.status}
        </span>

      </div>


      {/* FILE NAME */}

      <div className="flex items-center justify-between gap-3 mt-5">

        <h3
          className="font-semibold text-lg truncate"
          title={doc.filename}
        >
          {doc.filename}
        </h3>

        {approved && (
          <button
            onClick={() => navigate(`/ask/${doc.id}`)}
            className="shrink-0 bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-semibold transition shadow-lg shadow-blue-600/10"
          >
            Ask
          </button>
        )}

      </div>


      {/* VERSION AND CHUNKS */}

      <div className="grid grid-cols-2 gap-3 mt-5">

        <Info
          label="Version"
          value={doc.version || "v1.0"}
        />

        <Info
          label="Chunks"
          value={doc.chunks ?? 0}
        />

      </div>


      {/* ACTION BUTTONS */}

      <div className="flex items-center gap-3 mt-6">

        {!approved && (
          <div className="flex-1 text-sm text-slate-500 py-3">
            {doc.status === "Failed"
              ? "Processing failed"
              : "Document processing..."}
          </div>
        )}


        {/* DELETE ICON */}

        <button
          onClick={() => onDelete(doc.id, doc.filename)}
          disabled={deleting}
          title="Delete document"
          aria-label={`Delete ${doc.filename}`}
          className="w-12 h-12 flex items-center justify-center rounded-xl border border-red-500/30 text-red-400 hover:bg-red-500/10 hover:border-red-500/60 disabled:opacity-40 disabled:cursor-not-allowed transition"
        >

          {deleting ? (

            <span className="text-xs">
              ...
            </span>

          ) : (

            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M3 6h18" />
              <path d="M8 6V4h8v2" />
              <path d="M19 6l-1 14H6L5 6" />
              <path d="M10 11v6" />
              <path d="M14 11v6" />
            </svg>

          )}

        </button>

      </div>

    </div>
  );
}


/* =========================================================
   INFO
========================================================= */

function Info({ label, value }) {
  return (
    <div className="bg-black/20 rounded-xl p-3">

      <div className="text-xs text-slate-500">
        {label}
      </div>

      <div className="font-medium mt-1">
        {value}
      </div>

    </div>
  );
}


/* =========================================================
   ASK PAGE
========================================================= */

function AskPage({ documentId }) {
  const [document, setDocument] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);


  /* LOAD SELECTED DOCUMENT */

  useEffect(() => {
    fetch(`${API}/api/document/${documentId}`, {
      credentials: "include",
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          setDocument(data.document);
        }
      })
      .catch(console.error);

  }, [documentId]);


  /* ASK QUESTION */

  const askQuestion = async () => {
    if (!question.trim()) {
      return;
    }

    setLoading(true);
    setAnswer("");
    setSources([]);

    try {
      const response = await fetch(`${API}/api/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({
          document_id: Number(documentId),
          question,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        setAnswer(data.error || "Unable to generate answer.");
        return;
      }

      setAnswer(data.answer);
      setSources(data.sources || []);

    } catch (error) {
      console.error(error);
      setAnswer("Unable to connect to Askwise.");

    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="min-h-screen bg-[#050816] text-white">

      <header className="border-b border-white/10">

        <div className="max-w-6xl mx-auto px-6 py-5 flex items-center justify-between">

          <button
            onClick={() => navigate("/upload")}
            className="text-slate-400 hover:text-white"
          >
            ← Documents
          </button>

          <div className="font-bold text-xl">
            Ask<span className="text-blue-400">wise</span>
          </div>

          <ThemeToggle />

        </div>

      </header>


      <main className="max-w-5xl mx-auto px-6 py-10">

        {document && (

          <div className="mb-8">

            <div className="text-blue-400 text-sm">
              ASK DOCUMENT
            </div>

            <h1 className="text-3xl font-bold mt-2">
              {document.filename}
            </h1>

            <div className="flex gap-3 mt-4">

              <span className="px-3 py-1 rounded-full bg-blue-500/10 text-blue-400 text-xs">
                {document.version}
              </span>

              <span className="px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs">
                {document.status}
              </span>

            </div>

          </div>

        )}


        {/* QUESTION */}

        <div className="bg-white/[0.04] border border-white/10 rounded-3xl p-6">

          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask anything about this document..."
            rows={4}
            className="w-full bg-black/20 border border-white/10 rounded-2xl p-4 outline-none focus:border-blue-500 resize-none"
          />

          <button
            onClick={askQuestion}
            disabled={loading || !question.trim()}
            className="mt-4 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 px-6 py-3 rounded-xl font-semibold"
          >
            {loading ? "Thinking..." : "Ask Askwise →"}
          </button>

        </div>


        {/* ANSWER */}

        {answer && (

          <div className="mt-8 bg-white/[0.04] border border-white/10 rounded-3xl p-7">

            <div className="flex items-center justify-between gap-3 mb-3">
              <div className="text-blue-400 text-sm font-medium">
                AI ANSWER
              </div>
            </div>

            <div className="text-slate-200 leading-8 whitespace-pre-wrap">
              {answer}
            </div>

          </div>

        )}


        {/* SOURCES */}

        {sources.length > 0 && (

          <div className="mt-8">

            <h2 className="text-xl font-bold mb-4">
              Sources
            </h2>

            <div className="space-y-4">

              {sources.map((source, index) => (

                <div
                  key={index}
                  className="bg-white/[0.03] border border-white/10 rounded-2xl p-5"
                >

                  <div className="flex items-center justify-between gap-3 text-sm mb-2">
                    <span className="text-blue-400">
                      Chunk {source.chunk}
                    </span>
                  </div>

                  <p className="text-slate-400 text-sm leading-7">
                    {source.content}
                  </p>

                </div>

              ))}

            </div>

          </div>

        )}

      </main>

    </div>
  );
}


/* =========================================================
   APP ROUTER
========================================================= */

function App() {
  const [path, setPath] = useState(window.location.pathname);

  useEffect(() => {
    const handleNavigation = () => {
      setPath(window.location.pathname);
    };

    window.addEventListener("popstate", handleNavigation);

    return () => {
      window.removeEventListener("popstate", handleNavigation);
    };

  }, []);


  if (path.startsWith("/ask/")) {
    const documentId = path.split("/")[2];

    return <AskPage documentId={documentId} />;
  }

  if (path === "/upload") {
    return <UploadPage />;
  }

  return <LoginPage />;
}


export default App;
