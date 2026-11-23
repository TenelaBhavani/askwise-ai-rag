
from langchain_pinecone import PineconeVectorStore


class PineconeStore:

    def __init__(self, index_name, embedding):
        self.index_name = index_name
        self.embedding = embedding

    def create_store(self, docs):
        return PineconeVectorStore.from_documents(
            documents=docs,
            embedding=self.embedding,
            index_name=self.index_name
        )

    def connect(self):
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embedding
        )

    def delete_document(self, document_id):
        """
        Delete only the vectors belonging to one document.
        """
        store = self.connect()

        return store.delete(
            filter={
                "document_id": str(document_id)
            }
        )
