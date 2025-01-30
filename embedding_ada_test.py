from openai import OpenAI
from models import FindZebra
from numpy.linalg import norm
import numpy as np
from tqdm import tqdm
import os
from models import get_openai_token


def sectionize(content: str):
    material_sections = [
        sentence.strip()
        for sentence in content.replace("\n", ".").split(".")
        if sentence.strip()
    ]
    return material_sections


class Embedder:
    def __init__(self) -> None:
        self.tokens_used = 0
        self.client = OpenAI(api_key=get_openai_token())

    def get_embedding(self, content: str, max_retries=3):
        for _ in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    # input=content, model="text-embedding-ada-002", timeout=20
                    input=content,
                    model="text-embedding-3-large",
                    timeout=20,
                )
                break
            except Exception as e:
                print("Embedding error:", e)
                continue
        else:
            print("Embedding: all retries failed")
            return [0]
        self.tokens_used += response.usage.total_tokens

        embedding = response.data[0].embedding
        return embedding

    def get_embeddings(self, chunks: list[str], max_retries=3):

        try:
            response = self.client.embeddings.create(
                # input=content, model="text-embedding-ada-002", timeout=20
                input=chunks,
                model="text-embedding-3-large",
                timeout=20,
            )
        except Exception as e:
            return [0]
            print("Embedding error:", e)

        self.tokens_used += response.usage.total_tokens
        return [val.embedding for val in response.data]

    def get_embeddings_seq(self, chunks: list[str]):
        return [
            self.get_embedding(chunk)
            for chunk in tqdm(chunks, desc=f"Embed. {chunks[0][:20]}")
        ]


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def get_multiple_articles_concat(query: str, top_k: int):
    return "\n\n".join(
        [
            f"{title}:\n{material}"
            for title, material in fz.search_normalized_batch(query)[:top_k]
        ]
    )


class MaterialEmbedded:
    """
    Object for embedding articles with caching.
    TODO: will reload old embeddings even if chunking-strategy is changed
    """

    def __init__(self, material: str) -> None:
        self.material: str = material
        if not self._read_cache():
            self.chunks: list[str] = sectionize(material)
            self.embeddings = embedder.get_embeddings(self.chunks)
            self._write_cache()

    def __hash__(self):
        import hashlib
        import json

        # Use JSON serialization with sorted keys to create a consistent string representation
        serialized_material = json.dumps(self.material, sort_keys=True)
        # Use hashlib to generate a consistent hash across sessions
        return int(hashlib.sha256(serialized_material.encode("utf-8")).hexdigest(), 16)

    def _get_cache_path(self):
        import os

        if not os.path.exists("cache_embeddings"):
            os.makedirs("cache_embeddings")

        return os.path.join("cache_embeddings", f"large_{self.__hash__()}.pkl")

    def _write_cache(self):
        import os
        import pickle

        cache_path = self._get_cache_path()
        with open(cache_path, "wb") as cache_file:
            data_to_cache = {
                "material": self.material,
                "chunks": self.chunks,
                "embeddings": self.embeddings,
            }
            pickle.dump(data_to_cache, cache_file)
        print("Wrote to cache", cache_path)

    def _read_cache(self) -> bool:
        import os
        import pickle

        cache_path = self._get_cache_path()

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as cache_file:
                data_from_cache = pickle.load(cache_file)
                self.material = data_from_cache["material"]
                self.chunks = data_from_cache["chunks"]
                self.embeddings = data_from_cache["embeddings"]
            print("Read from cache", cache_path)
            return True
        else:
            print("No cache found", cache_path)
            return False

    def get_knn(self, target_embedding, top_k=5) -> list[str]:
        distances: list[float] = [
            cosine_similarity(target_embedding, embedding)
            for embedding in self.embeddings
        ]
        top_indices = np.argsort(distances)[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]


embedder = Embedder()
fz = FindZebra()

# print(embedder.get_embeddings(["Oh hello", "Morning"]))
