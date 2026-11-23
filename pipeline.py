from pathlib import Path

from prompts.rag_prompt import PROMPT


class RAGPipeline:

    def __init__(self, retriever, llm):

        self.retriever = retriever
        self.llm = llm

    def ask(self, question):
        docs = self.retriever.retrieve(question)

        if not docs:
            return "I couldn't find relevant information in the indexed PDFs."

        source_ids = {}
        context_sections = []
        for doc in docs:
            metadata = doc.metadata
            source = Path(str(metadata.get("source", "unknown PDF"))).name
            page = metadata.get("page")
            page_number = page + 1 if isinstance(page, int) else None
            source_key = (source, page_number)

            if source_key not in source_ids:
                source_ids[source_key] = len(source_ids) + 1

            source_id = source_ids[source_key]
            page_label = f", page {page_number}" if page_number is not None else ""
            context_sections.append(
                f"[Source {source_id}: {source}{page_label}]\n{doc.page_content}"
            )

        context = "\n\n".join(context_sections)

        prompt = PROMPT.format(
            context=context,
            question=question
        )

        response = self.llm.invoke(prompt)
        answer = response if isinstance(response, str) else response.content
        sources = "; ".join(
            f"[{source_id}] {source}"
            + (f", page {page}" if page is not None else "")
            for (source, page), source_id in source_ids.items()
        )
        return f"{answer.rstrip()}\n\nSources: {sources}"