import faiss
import numpy as np


class VectorStore:

    def __init__(self):

        self.index = None
        self.documents = []

    def build(
        self,
        documents: list[str],
        embeddings: list[list[float]],
    ):

        if not embeddings:

            raise ValueError(
                "No embeddings were provided."
            )

        vectors = np.array(
            embeddings,
            dtype="float32"
        )

        dimension = vectors.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(vectors)

        self.documents = documents

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[str]:

        if self.index is None:
            return []

        if not self.documents:
            return []

        query = np.array(
            [query_embedding],
            dtype="float32"
        )

        k = min(
            top_k,
            len(self.documents)
        )

        distances, indices = (
            self.index.search(query, k)
        )

        results = []

        for index in indices[0]:

            if index >= 0:

                results.append(
                    self.documents[index]
                )

        return results
