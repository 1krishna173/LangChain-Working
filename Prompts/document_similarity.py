from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell Me about the Rohit Sharma"


doc_embed = embedding.embed_documents(documents)
query_em = embedding.embed_query(query)


similarity_scores = cosine_similarity([query_em], doc_embed)[0]

for doc, score in zip(documents, similarity_scores):
    print(f"[{score:.4f}] {doc}")

best_idx = np.argmax(similarity_scores)
print(f"\nMost relevant document:\n{documents[best_idx]}")