from rag.embeddings import (
    create_embeddings,
    create_query_embedding,
    split_text,
)

from rag.vector_store import VectorStore

from config.settings import TOP_K


class DocumentRetriever:

    def __init__(self):

        self.vector_store = VectorStore()

        self.chunks = []

        self.ready = False

    def process_document(
        self,
        text: str
    ):

        self.chunks = split_text(text)

        if not self.chunks:

            raise ValueError(
                "No readable text was found."
            )

        embeddings = create_embeddings(
            self.chunks
        )

        self.vector_store.build(
            self.chunks,
            embeddings
        )

        self.ready = True

    def search(
        self,
        question: str
    ) -> list[str]:

        if not self.ready:
            return []

        query_embedding = (
            create_query_embedding(question)
        )

        return self.vector_store.search(
            query_embedding,
            TOP_K
        )

    def get_context(
        self,
        question: str
    ) -> str:

        results = self.search(question)

        if not results:
            return ""

        return "\n\n---\n\n".join(
            results
        )
