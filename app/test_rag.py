# test_rag.py
from rag_pipeline import process_url_for_rag

print("Start")
pkg = process_url_for_rag(
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    max_tokens_per_chunk=256,
    embed_backend="sbert",
)
print("OK", pkg["title"], len(pkg["records"]))
