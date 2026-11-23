import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from pipeline import RAGPipeline


class RAGPipelineTests(unittest.TestCase):
    def test_does_not_generate_answer_without_relevant_pdf_content(self):
        llm = Mock()
        pipeline = RAGPipeline(Mock(retrieve=Mock(return_value=[])), llm)

        answer = pipeline.ask("What does the PDF say?")

        self.assertIn("indexed PDFs", answer)
        llm.invoke.assert_not_called()

    def test_answer_includes_pdf_names_and_one_based_page_sources(self):
        docs = [
            SimpleNamespace(
                metadata={"source": "/library/guide.pdf", "page": 0},
                page_content="First excerpt.",
            ),
            SimpleNamespace(
                metadata={"source": "/library/guide.pdf", "page": 0},
                page_content="Another excerpt from this page.",
            ),
            SimpleNamespace(
                metadata={"source": "/library/guide.pdf", "page": 2},
                page_content="A later excerpt.",
            ),
        ]
        retriever = Mock()
        retriever.retrieve.return_value = docs
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content="PDF-grounded answer.")

        answer = RAGPipeline(retriever, llm).ask("Summarize the guide.")

        self.assertEqual(
            answer,
            "PDF-grounded answer.\n\nSources: [1] guide.pdf, page 1; "
            "[2] guide.pdf, page 3",
        )
        prompt = llm.invoke.call_args.args[0]
        self.assertIn("[Source 1: guide.pdf, page 1]", prompt)
        self.assertIn("[Source 2: guide.pdf, page 3]", prompt)


if __name__ == "__main__":
    unittest.main()
