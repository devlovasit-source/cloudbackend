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

        # 🔥 COLLECTIONS (CLEANED)
        self.collection = os.getenv("QDRANT_COLLECTION", "wardrobe")
        self.memory_collection = os.getenv("QDRANT_MEMORY_COLLECTION", "outfit_memory")
        self.user_memory_collection = os.getenv("QDRANT_USER_MEMORY_COLLECTION", "user_memory")

        # 🔥 HYBRID VECTOR (512)
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

    def enabled(self):
        return bool(self.client)

    # =========================
    # UPSERT (MAIN)
    # =========================
    def upsert_item(self, item_id, vector, payload):
        if not self._ensure():
            return

        if not vector or len(vector) != self.vector_size:
            print("❌ Vector size mismatch:", len(vector))
            return

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
            print("✅ QDRANT UPSERT:", item_id)

        except Exception as e:
            print("❌ Upsert failed:", str(e))

    # =========================
    # WARDROBE HELPER
    # =========================
    def upsert_wardrobe_item(self, item: dict):
        try:
            self.upsert_item(
                item_id=item["id"],
                vector=item.get("embedding", [0.0] * self.vector_size),
                payload={
                    "userId": item.get("userId"),
                    "category": item.get("category"),
                    "sub_category": item.get("sub_category"),
                    "color": item.get("color_code"),
                    "image_url": item.get("image_url"),
                    "pixel_hash": item.get("pixel_hash"),
                }
            )
        except Exception as e:
            print("❌ Wardrobe upsert failed:", str(e))

    # =========================
    # STYLE BOARD UPSERT
    # =========================
    def upsert_style_board(self, board_id, vector, payload):
        if not self._ensure():
            return

        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=f"board_{board_id}",
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except Exception as e:
            print("❌ Board upsert failed:", str(e))

    # =========================
    # SEARCH (FILTERED)
    # =========================
    def search_similar(self, vector, user_id, category=None, limit=5):
        if not self._ensure() or not vector:
            return []

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
            ]

        except Exception as e:
            print("❌ Search failed:", str(e))
            return []

    # =========================
    # SEMANTIC RETRIEVE
    # =========================
    def semantic_retrieve(self, vector, user_id, limit=40):
        return self.search_similar(vector, user_id, None, limit)

    # =========================
    # BOARD SEARCH
    # =========================
    def search_similar_boards(self, vector, limit=10):
        if not self._ensure():
            return []

        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=limit,
            )

            return [
                {
                    "id": str(r.id),
                    "score": float(r.score),
                    "payload": r.payload or {},
                }
                for r in results
                if str(r.id).startswith("board_")
            ]

        except Exception as e:
            print("❌ Board search failed:", str(e))
            return []

    # =========================
    # SCROLL BOARDS
    # =========================
    def get_all_boards(self, limit=100):
        if not self._ensure():
            return []

        try:
            response = self.client.scroll(
                collection_name=self.collection,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )

            points = response[0] if isinstance(response, tuple) else []

            return [
                {
                    "id": str(p.id),
                    "embedding": getattr(p, "vector", None),
                    "payload": getattr(p, "payload", {}) or {},
                }
                for p in points
                if str(p.id).startswith("board_")
            ]

        except Exception as e:
            print("❌ Fetch boards failed:", str(e))
            return []

    # =========================
    # FEEDBACK
    # =========================
    def update_feedback(self, item_id, feedback):
        if not self._ensure():
            return

        try:
            self.client.set_payload(
                collection_name=self.collection,
                payload={"feedback": feedback},
                points=[item_id],
            )
        except Exception as e:
            print("❌ Feedback update failed:", str(e))

    # =========================
    # MEMORY VECTOR
    # =========================
    def upsert_memory_vector(self, point_id, vector, payload):
        if not self._ensure():
            return

        try:
            self.client.upsert(
                collection_name=self.memory_collection,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
            )
        except Exception as e:
            print("❌ Memory upsert failed:", str(e))

    # =========================
    # USER MEMORY
    # =========================
    def upsert_user_memory(self, user_id, vector, payload):
        if not self._ensure():
            return None

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
    # 🔥 PIXEL DUPLICATE (IMPORTANT)
    # =========================
    def find_pixel_duplicate(self, user_id, pixel_hash, max_distance=6):
        if not self._ensure() or not pixel_hash:
            return {"checked": False, "is_duplicate": False}

        try:
            query_filter = Filter(
                must=[FieldCondition(key="userId", match=MatchValue(value=str(user_id)))]
            )

            response = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=200,
                with_payload=True,
            )

            points = response[0] if isinstance(response, tuple) else []

            for point in points:
                payload = point.payload or {}
                candidate_hash = payload.get("pixel_hash")

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
