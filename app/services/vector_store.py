import chromadb
import hashlib
from typing import List, Optional
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from app.core.config import settings
from app.models.schemas import ProductInfo

class LightweightEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        results = []
        for doc in input:
            vec = [0.0] * 128
            tokens = doc.lower().split()
            for token in tokens:
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                vec[h % 128] += 1.0
            norm = sum(x * x for x in vec) ** 0.5
            results.append([x / (norm or 1.0) for x in vec])
        return results

class CatalogStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.embedding_fn = LightweightEmbeddingFunction()

    def _get_collection(self, brand_id: str):
        safe_brand = "".join(c if c.isalnum() else "_" for c in brand_id).lower()
        collection_name = f"catalog_{safe_brand}"[:63]
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_product(self, brand_id: str, product: ProductInfo):
        collection = self._get_collection(brand_id)
        doc_text = f"SKU: {product.sku} | Title: {product.title} | Price: {product.currency} {product.price} | Sizes: {', '.join(product.sizes)} | In Stock: {product.in_stock} | Description: {product.description}"
        
        metadata = {
            "sku": product.sku,
            "title": product.title,
            "price": float(product.price),
            "currency": product.currency,
            "in_stock": int(product.in_stock),
            "stock_quantity": product.stock_quantity,
            "sizes": ", ".join(product.sizes),
            "colors": ", ".join(product.colors),
            "checkout_url": product.checkout_url
        }

        collection.upsert(
            documents=[doc_text],
            metadatas=[metadata],
            ids=[product.sku]
        )

    def get_product_by_sku(self, brand_id: str, sku: str) -> Optional[ProductInfo]:
        collection = self._get_collection(brand_id)
        res = collection.get(ids=[sku])
        if res and res["metadatas"] and len(res["metadatas"]) > 0:
            m = res["metadatas"][0]
            return ProductInfo(
                sku=m["sku"],
                title=m["title"],
                description="",
                price=float(m["price"]),
                currency=m.get("currency", "INR"),
                in_stock=bool(m.get("in_stock", 1)),
                stock_quantity=int(m.get("stock_quantity", 10)),
                sizes=[s.strip() for s in m.get("sizes", "").split(",") if s.strip()],
                colors=[c.strip() for c in m.get("colors", "").split(",") if c.strip()],
                checkout_url=m.get("checkout_url", "")
            )
        return None

    def search_products(self, brand_id: str, query: str, limit: int = 2) -> List[ProductInfo]:
        collection = self._get_collection(brand_id)
        if collection.count() == 0:
            return []
            
        results = collection.query(
            query_texts=[query],
            n_results=min(limit, collection.count())
        )
        
        products = []
        if results and results["metadatas"] and len(results["metadatas"]) > 0:
            for m in results["metadatas"][0]:
                products.append(
                    ProductInfo(
                        sku=m["sku"],
                        title=m["title"],
                        description="",
                        price=float(m["price"]),
                        currency=m.get("currency", "INR"),
                        in_stock=bool(m.get("in_stock", 1)),
                        stock_quantity=int(m.get("stock_quantity", 10)),
                        sizes=[s.strip() for s in m.get("sizes", "").split(",") if s.strip()],
                        colors=[c.strip() for c in m.get("colors", "").split(",") if c.strip()],
                        checkout_url=m.get("checkout_url", "")
                    )
                )
        return products

catalog_store = CatalogStore()
