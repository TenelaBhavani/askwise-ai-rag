Askwise PDF RAG
===============

This project answers questions from PDF files indexed in Pinecone. It does not
use general knowledge as a fallback when a question cannot be answered from
the PDFs.

PDF workflow
------------

1. Put PDF files in data/raw/.
2. Set PINECONE_API_KEY, PINECONE_INDEX, and GROQ_API_KEY in .env.
3. Run `python ingest.py` to load, split, embed, and index the PDFs.
4. Run `python query.py` for the terminal interface or `python app.py` for the
   web interface.

Answers include source PDF names and page numbers when the loader provides page
metadata. If retrieval finds no relevant PDF content, the system says it cannot
answer from the indexed PDFs instead of generating an unsupported response.

Pipeline components
-------------------

| Module                    | Responsibility                                  |
| ------------------------- | ----------------------------------------------- |
| loaders/pdf_loader.py     | Load PDF pages into `Document` objects          |
| ingest.py                 | Load PDFs from data/raw/ and populate Pinecone  |
| splitters/text_splitter.py| Split PDF pages into overlapping chunks         |
| embeddings/sentence_transformer.py | Generate embeddings                  |
| vectorstores/pinecone_store.py | Create and query the PDF vector index       |
| retrievers/retriever.py   | Retrieve PDF chunks relevant to a question      |
| prompts/rag_prompt.py     | Require answers grounded in PDF excerpts        |
| pipeline.py               | Build grounded answers and report PDF sources   |
| query.py                  | Ask questions against the indexed PDFs         |
| app.py                    | PDF-focused web interface                      |

Default retrieval settings: chunk size 500, overlap 50, top-k 5.
