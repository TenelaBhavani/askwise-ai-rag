# retrievers/retriever.py

from config import TOP_K


class Retriever:

    def __init__(self, vectorstore):

        self.retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": TOP_K
            }
        )

    def retrieve(self, question):

        try:
            documents = self.retriever.invoke(question)

            return documents

        except Exception as e:

            print("Retriever error:", str(e))

            raise