import os
import time
import uuid
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from services.image_fingerprint import hamming_distance_hex

load_dotenv()


class QdrantService:

    def __init__(self):
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")

        self.collection = os.getenv("QDRANT_COLLECTION", "wardrobe")
        self.memory_collection = os.getenv("QDRANT_MEMORY_COLLECTION", "outfit_memory")
        self.user_memory_collection = os.getenv("QDRANT_USER_MEMORY_COLLECTION", "user_memory")

        self.vector_size = 512
        self.memory_vector_size = int(os.getenv("QDRANT_MEMORY_VECTOR_SIZE", "8"))
        self.user_memory_vector_size = self.vector_size

        self.client = None
        self._initialized = False

        if self.url:
            try:
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
            except Exception as e:
                print("❌ Qdrant init failed:", str(e))

    # =========================
    # INIT
    # =========================
    def init(self):
        if not self.client or self._initialized:
            return

        print("Initializing Qdrant...")

        self._create_collection(self.collection, self.vector_size)
        self._create_collection(self.memory_collection, self.memory_vector_size)
        self._create_collection(self.user_memory_collection, self.user_memory_vector_size)

        self._initialized = True

    def _create_collection(self, name, size):
        try:
            existing = [c.name for c in self.client.get_collections().collections]

            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=size, distance=Distance.COSINE),
                )
                print(f"✅ Created collection: {name}")

        except Exception as e:
            print("❌ Collection init error:", str(e))

    def _ensure(self):
        if not self.client:
            return False
        if not self._initialized:
            self.init()
        return True

    # =========================
    # 🔥 NORMALIZATION
    # =========================
    def _normalize(self, vector):
        if not vector:
            return vector

        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0:
            return vector

        return [v / norm for v in vector]

    # =========================
    # UPSERT (MAIN)
    # =========================
    def upsert_item(self, item_id, vector, payload):
        if not self._ensure():
            return

        if not vector or len(vector) != self.vector_size:
            print("❌ Vector size mismatch:", len(vector))
            return

        vector = self._normalize(vector)

        payload["timestamp"] = time.time()

        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=item_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except Exception as e:
            print("❌ Upsert failed:", str(e))

    # =========================
    # 🔥 BATCH UPSERT (NEW)
    # =========================
    def upsert_batch(self, items):
        if not self._ensure():
            return

        points = []

        for item in items:
            vec = item.get("vector")
            if not vec or len(vec) != self.vector_size:
                continue

            vec = self._normalize(vec)

            payload = item.get("payload", {})
            payload["timestamp"] = time.time()

            points.append(
                PointStruct(
                    id=item.get("id", str(uuid.uuid4())),
                    vector=vec,
                    payload=payload,
                )
            )

        if not points:
            return

        try:
            self.client.upsert(
                collection_name=self.collection,
                points=points,
            )
        except Exception as e:
            print("❌ Batch upsert failed:", str(e))

    # =========================
    # 🔥 SEARCH (WITH THRESHOLD)
    # =========================
    def search_similar(self, vector, user_id, category=None, limit=5, score_threshold=0.6):
        if not self._ensure() or not vector:
            return []

        vector = self._normalize(vector)

        try:
            must = [
                FieldCondition(key="userId", match=MatchValue(value=str(user_id)))
            ]

            if category:
                must.append(FieldCondition(key="category", match=MatchValue(value=category)))

            query_filter = Filter(must=must)

            results = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=limit,
                query_filter=query_filter,
            )

            return [
                {
                    "id": str(r.id),
                    "score": float(r.score),
                    "payload": r.payload or {},
                }
                for r in results
                if r.score >= score_threshold
            ]

        except Exception as e:
            print("❌ Search failed:", str(e))
            return []

    # =========================
    # 🔥 FAST PIXEL DUPLICATE
    # =========================
    def find_pixel_duplicate(self, user_id, pixel_hash, max_distance=6):
        if not self._ensure() or not pixel_hash:
            return {"checked": False, "is_duplicate": False}

        try:
            # 🔥 only fetch limited candidates
            response = self.client.scroll(
                collection_name=self.collection,
                limit=100,
                with_payload=True,
                scroll_filter=Filter(
                    must=[FieldCondition(key="userId", match=MatchValue(value=str(user_id)))]
                ),
            )

            points = response[0] if isinstance(response, tuple) else []

            for point in points:
                candidate_hash = (point.payload or {}).get("pixel_hash")
                if not candidate_hash:
                    continue

                dist = hamming_distance_hex(pixel_hash, candidate_hash)

                if dist is not None and dist <= max_distance:
                    return {
                        "checked": True,
                        "is_duplicate": True,
                        "id": str(point.id),
                        "distance": dist,
                    }

            return {"checked": True, "is_duplicate": False}

        except Exception:
            return {"checked": True, "is_duplicate": False}

    # =========================
    # USER MEMORY
    # =========================
    def upsert_user_memory(self, user_id, vector, payload):
        if not self._ensure() or not vector:
            return None

        vector = self._normalize(vector)

        try:
            self.client.upsert(
                collection_name=self.user_memory_collection,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except Exception as e:
            print("❌ User memory upsert failed:", str(e))

    # =========================
    # STATUS
    # =========================
    def status(self):
        return {
            "enabled": bool(self.client),
            "initialized": self._initialized,
            "collection": self.collection,
            "vector_size": self.vector_size,
        }


# =========================
# SINGLETON
# =========================
qdrant_service = QdrantService()
