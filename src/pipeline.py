from src.rag.retriever import ComplaintRetriever
from src.rag.prompt import build_prompt
from src.rag.generator import AnswerGenerator


class RAGPipeline:
    def __init__(self):
        self.retriever = ComplaintRetriever()
        self.generator = AnswerGenerator()

    def ask(self, question, k=5):
        docs, metadata = self.retriever.retrieve(question, k=k)

        context = "\n\n".join(docs)
        prompt = build_prompt(context, question)

        answer = self.generator.generate(prompt)

        return {
            "answer": answer,
            "sources": metadata[:2]  # show top 1–2 sources
        }
