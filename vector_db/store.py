# ============================================================
#  JanAI — Vector Database (ChromaDB)
#  Stores complaints as semantic embeddings
#  Enables similarity search & recurrence detection
# ============================================================

import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
import uuid

# ── Initialize ChromaDB ──────────────────────────────────────
# PersistentClient saves data to disk — survives restarts
client = chromadb.PersistentClient(path="./janai_vectordb")

collection = client.get_or_create_collection(
    name     = "complaints",
    metadata = {"hnsw:space": "cosine"}   # cosine similarity search
)

# ── Embedding Model ──────────────────────────────────────────
# Converts text → 384-dimensional vector
# Fast, lightweight, works offline
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ── Add Complaint ────────────────────────────────────────────
def add_complaint(
    text:     str,
    category: str,
    severity: str,
    location: str
) -> str:
    """
    Embed and store a new complaint in the vector database.

    Returns
    -------
    complaint_id : unique UUID string
    """
    complaint_id = str(uuid.uuid4())
    embedding    = embedder.encode(text).tolist()

    collection.add(
        ids        = [complaint_id],
        embeddings = [embedding],
        documents  = [text],
        metadatas  = [{
            "category":  category,
            "severity":  severity,
            "location":  location,
            "timestamp": datetime.now().isoformat(),
            "status":    "OPEN"
        }]
    )

    print(f"[VectorDB] Stored complaint: {complaint_id[:8]}...")
    return complaint_id


# ── Similarity Search ────────────────────────────────────────
def find_similar_complaints(text: str, top_k: int = 5) -> list:
    """
    Find complaints semantically similar to the given text.
    Used to detect patterns and recurring issues.

    Returns
    -------
    list of dicts: complaint text, similarity score, metadata
    """
    embedding = embedder.encode(text).tolist()

    results = collection.query(
        query_embeddings = [embedding],
        n_results        = min(top_k, collection.count() or 1)
    )

    similar = []
    for i, doc in enumerate(results["documents"][0]):
        similar.append({
            "complaint":  doc,
            "similarity": round(1 - results["distances"][0][i], 3),
            "metadata":   results["metadatas"][0][i]
        })

    return similar


# ── Recurrence Count ─────────────────────────────────────────
def get_recurrence_count(
    text:      str,
    threshold: float = 0.80
) -> int:
    """
    Count how many similar complaints exist above similarity threshold.
    Used directly in the JanAI priority scoring formula.

    threshold=0.80 means 80% similar — same issue, different wording
    """
    if collection.count() == 0:
        return 0

    similar = find_similar_complaints(text, top_k=20)
    count   = sum(1 for s in similar if s["similarity"] >= threshold)
    return count


# ── Update Status ────────────────────────────────────────────
def update_complaint_status(complaint_id: str, status: str) -> bool:
    """
    Update the status of a complaint.
    Status options: OPEN | IN_PROGRESS | RESOLVED | VERIFIED
    """
    try:
        collection.update(
            ids       = [complaint_id],
            metadatas = [{"status": status, "updated_at": datetime.now().isoformat()}]
        )
        return True
    except Exception as e:
        print(f"[VectorDB] Update failed: {e}")
        return False


# ── Get All Complaints ────────────────────────────────────────
def get_all_complaints() -> list:
    """Fetch all stored complaints — used for dashboard."""
    results = collection.get()
    complaints = []
    for i, doc in enumerate(results["documents"]):
        complaints.append({
            "id":       results["ids"][i],
            "text":     doc,
            "metadata": results["metadatas"][i]
        })
    return complaints


# ── Quick Test ───────────────────────────────────────────────
if __name__ == "__main__":
    # Add sample data
    print("Adding sample complaints...")
    id1 = add_complaint("Big pothole on MG Road near bus stop", "roads",      "HIGH",   "MG Road")
    id2 = add_complaint("No water supply in Sector 5 for 2 days",  "water",   "MEDIUM", "Sector 5")
    id3 = add_complaint("Road is badly damaged near MG Road signal", "roads", "MEDIUM", "MG Road")
    id4 = add_complaint("Garbage not collected in Sector 5",    "sanitation",  "LOW",    "Sector 5")

    # Similarity search
    print("\nSimilar to 'broken road MG Road':")
    for r in find_similar_complaints("broken road MG Road", top_k=3):
        print(f"  [{r['similarity']}] {r['complaint']}")

    # Recurrence count
    count = get_recurrence_count("pothole on MG Road")
    print(f"\nRecurrence count for road issue: {count}")

    # All complaints
    print(f"\nTotal stored complaints: {len(get_all_complaints())}")
