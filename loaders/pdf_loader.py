# loaders/pdf_loader.py

from langchain_community.document_loaders import PyPDFLoader

class PDFLoader:

    def load(self, pdf_path):

        loader = PyPDFLoader(pdf_path)

        return loader.load()