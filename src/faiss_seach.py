from sentence_transformers import SentenceTransformer
import faiss
from hatesonar import Sonar

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
index = faiss.read_index('index_offensive_sentences')


def search(query):
    query_vector = model.encode([query])
    k = 5
    top_k = index.search(query_vector, k)
    min_dist = top_k[0].min()
    if min_dist < 0.1:
        return 1
    return 0
