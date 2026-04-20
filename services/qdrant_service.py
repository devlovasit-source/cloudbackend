import os
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

# Load env
load_dotenv()


class QdrantService:

    def __init__(self):
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")

        # 🔥 LOCKED COLLECTIONS
        self.collection = os.getenv("QDRANT_COLLECTION", "wardrobe")
        self.memory_collection = os.getenv("QDRANT_MEMORY_COLLECTION", "outfit_memory")
        self.image_collection = os.getenv("QDRANT_IMAGE_COLLECTION", "wardrobe_image")

        self.vector_size = 384
        self.memory_vector_size = 8
        self.image_vector_size = int(os.getenv("QDRANT_IMAGE_VECTOR_SIZE", "512"))

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
        self._create_collection(self.image_collection, self.image_vector_size)

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
    # 🔥 MAIN UPSERT (GENERIC)
    # =========================
    def upsert_item(self, item_id, vector, payload):
        if not self._ensure():
            return

        try:
            if not payload.get("userId"):
                print("⚠️ Missing userId in payload")

            self.client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=item_id,
                        vector=vector,
                        payload=payload
                    )
                ],
            )

            print("✅ QDRANT UPSERT:", item_id)

        except Exception as e:
            print("❌ Upsert item failed:", str(e))

    # =========================
    # 🔥 WARDROBE HELPER (NEW)
    # =========================
    def upsert_wardrobe_item(self, item: dict):
        """
        Backward compatible helper
        Allows both styles:
        - upsert_item(...)
        - upsert_wardrobe_item({...})
        """

        try:
            self.upsert_item(
                item_id=item["id"],
                vector=item.get("embedding", [0.0] * self.vector_size),
                payload={
                    "userId": item.get("userId"),
                    "type": item.get("type"),
                    "category": item.get("category"),
                    "color": item.get("color"),
                    "image_url": item.get("image_url"),
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
    # SEARCH
    # =========================
    def search_similar(self, vector, user_id, limit=5):
        if not self._ensure():
            return []

    def semantic_retrieve(self, query_vector, user_id, limit=40):
        if not self._ensure():
            return []
        try:
            query_filter = Filter(
                must=[FieldCondition(key="userId", match=MatchValue(value=str(user_id)))]
            )
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=max(1, int(limit)),
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
        except Exception:
            return []

        try:
            query_filter = Filter(
                must=[FieldCondition(key="userId", match=MatchValue(value=str(user_id)))]
            )
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
    # SCROLL ALL BOARDS
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
    # IMAGE VECTOR
    # =========================
    def upsert_image_vector(self, point_id, vector, payload):
        if not self._ensure():
            return

        try:
            self.client.upsert(
                collection_name=self.image_collection,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
            )
        except Exception as e:
            print("❌ Image upsert failed:", str(e))

    def find_image_duplicate(self, image_vector, user_id, threshold=0.985):
        if not self._ensure() or not image_vector:
            return {"checked": False, "is_duplicate": False, "id": None, "score": 0.0}
        try:
            query_filter = Filter(
                must=[FieldCondition(key="userId", match=MatchValue(value=str(user_id)))]
            )
            results = self.client.search(
                collection_name=self.image_collection,
                query_vector=image_vector,
                limit=1,
                query_filter=query_filter,
            )
            if not results:
                return {"checked": True, "is_duplicate": False, "id": None, "score": 0.0}
            top = results[0]
            score = float(getattr(top, "score", 0.0) or 0.0)
            return {
                "checked": True,
                "is_duplicate": score >= float(threshold or 0.985),
                "id": str(getattr(top, "id", "") or "") or None,
                "score": score,
            }
        except Exception:
            return {"checked": True, "is_duplicate": False, "id": None, "score": 0.0}

    def find_pixel_duplicate(self, user_id, pixel_hash, max_distance=6):
        if not self._ensure() or not pixel_hash:
            return {"checked": False, "is_duplicate": False, "id": None, "distance": None}
        try:
            query_filter = Filter(
                must=[FieldCondition(key="userId", match=MatchValue(value=str(user_id)))]
            )
            response = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=200,
                with_payload=True,
                with_vectors=False,
            )
            points = response[0] if isinstance(response, tuple) else []
            best_id = None
            best_distance = None
            for point in points or []:
                payload = getattr(point, "payload", {}) or {}
                candidate_hash = str(payload.get("pixel_hash") or "")
                if not candidate_hash:
                    continue
                distance = hamming_distance_hex(pixel_hash, candidate_hash)
                if distance is None:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_id = str(getattr(point, "id", "") or "") or None
            return {
                "checked": True,
                "is_duplicate": best_distance is not None and int(best_distance) <= int(max_distance),
                "id": best_id,
                "distance": best_distance,
            }
        except Exception:
            return {"checked": True, "is_duplicate": False, "id": None, "distance": None}

    # =========================
    # COSINE SIMILARITY
    # =========================
    @staticmethod
    def cosine_similarity(vec1, vec2):
        try:
            import numpy as np

            v1 = np.array(vec1)
            v2 = np.array(vec2)

            if v1.size == 0 or v2.size == 0:
                return 0.0

            return float(
                np.dot(v1, v2)
                / (np.linalg.norm(v1) * np.linalg.norm(v2))
            )
        except Exception:
            return 0.0

    # =========================
    # STATUS
    # =========================
    def status(self):
        return {
            "enabled": bool(self.client),
            "initialized": self._initialized,
            "collection": self.collection,
            "memory_collection": self.memory_collection,
            "image_collection": self.image_collection,
        }


# =========================
# SINGLETON
# =========================
qdrant_service = QdrantService()
