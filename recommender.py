"""
Day 4 - Customer Recommendation Engine
Squad Python Specialization - Data Science, AI & Analytics

"Customers like you also bought": purchase histories are sets, similarity is
Jaccard, and recommendations are ranked by an explainable scoring rule.

Core Python only (csv, pathlib). No ML libraries.
"""
from __future__ import annotations

import csv
from pathlib import Path

DATA_PATH = Path("data") / "purchases.csv"
OUTPUT_PATH = Path("output") / "recommendations.csv"

# Weight of the popularity signal in the ranking rule (see README).
POPULARITY_WEIGHT = 0.25

CSV_COLUMNS = [
    "customer_id", "customer_name", "rank", "product_id", "product_name",
    "category", "score", "supporting_customers",
]


class UnknownCustomerError(KeyError):
    """Raised when a customer ID is not in the engine's purchase history."""

    def __str__(self) -> str:  # KeyError normally prints the repr of its argument
        return str(self.args[0]) if self.args else "unknown customer"


# --------------------------------------------------------------------------- #
# Step 1: reading and cleaning
# --------------------------------------------------------------------------- #
def _normalise_id(value: str | None) -> str:
    """Strip whitespace and upper-case an ID so 'c106 ' becomes 'C106'."""
    return (value or "").strip().upper()


def read_purchases(path: Path = DATA_PATH) -> tuple[list[dict], list[dict]]:
    """Return (valid_rows, rejected_rows).

    Cleaning rules (each rejected row gets a 'reject_reason'):
      1. IDs are normalised (strip + upper-case): 'c106' -> 'C106'. Kept, not rejected.
      2. Missing customer_id or product_id (blank)      -> rejected 'missing_customer_id' / 'missing_product_id'.
      3. quantity that is not a whole number, or <= 0   -> rejected 'invalid_quantity'.
      4. A purchase_id already seen (repeated row)      -> rejected 'duplicate_purchase_id'
         (the first occurrence is kept).
    Names/categories are stripped of surrounding whitespace.
    """
    valid: list[dict] = []
    rejected: list[dict] = []
    seen_purchase_ids: set[str] = set()

    with open(path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {key: (value or "").strip() for key, value in raw.items()}
            row["customer_id"] = _normalise_id(row.get("customer_id"))
            row["product_id"] = _normalise_id(row.get("product_id"))
            row["purchase_id"] = _normalise_id(row.get("purchase_id"))

            reason = None
            if not row["customer_id"]:
                reason = "missing_customer_id"
            elif not row["product_id"]:
                reason = "missing_product_id"
            else:
                try:
                    quantity = int(row.get("quantity", ""))
                except ValueError:
                    quantity = 0
                if quantity <= 0:
                    reason = "invalid_quantity"
                else:
                    row["quantity"] = quantity
            if reason is None and row["purchase_id"] in seen_purchase_ids:
                reason = "duplicate_purchase_id"

            if reason:
                rejected.append({**raw, "reject_reason": reason})
            else:
                seen_purchase_ids.add(row["purchase_id"])
                valid.append(row)

    return valid, rejected


# --------------------------------------------------------------------------- #
# Step 3: similarity
# --------------------------------------------------------------------------- #
def jaccard(a: set, b: set) -> float:
    """|A ∩ B| / |A ∪ B|.  Two empty sets have no evidence of similarity -> 0.0."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class RecommendationEngine:
    """Recommends products a customer has not bought, based on similar customers."""

    def __init__(self, purchases: list[dict], min_similarity: float = 0.0) -> None:
        self.min_similarity = min_similarity

        # All structures are built once here; no later method touches raw rows.
        self._histories: dict[str, set[str]] = {}
        self._customer_names: dict[str, str] = {}
        self._products: dict[str, dict[str, str]] = {}
        self._buyers: dict[str, set[str]] = {}  # product -> customers who bought it

        for row in purchases:
            cid, pid = row["customer_id"], row["product_id"]
            self._histories.setdefault(cid, set()).add(pid)
            self._customer_names.setdefault(cid, row.get("customer_name", ""))
            self._buyers.setdefault(pid, set()).add(cid)
            info = self._products.setdefault(pid, {"name": "", "category": ""})
            # Fill from the first row that has a value (a later row may fix a blank).
            info["name"] = info["name"] or row.get("product_name", "")
            info["category"] = info["category"] or row.get("category", "")

        # Popularity = number of DISTINCT customers who bought the product, so one
        # customer buying a product five times cannot inflate it.
        self._popularity: dict[str, int] = {p: len(c) for p, c in self._buyers.items()}

    # -- simple lookups ---------------------------------------------------- #
    @property
    def customers(self) -> list[str]:
        return sorted(self._histories)

    @property
    def products(self) -> list[str]:
        return sorted(self._products)

    def _require(self, customer_id: str) -> None:
        if customer_id not in self._histories:
            raise UnknownCustomerError(f"Unknown customer ID: {customer_id!r}")

    def customer_name(self, customer_id: str) -> str:
        self._require(customer_id)
        return self._customer_names[customer_id]

    def product_info(self, product_id: str) -> dict[str, str]:
        return self._products[product_id]

    def popularity(self, product_id: str) -> int:
        return self._popularity[product_id]

    def history(self, customer_id: str) -> set[str]:
        self._require(customer_id)
        return set(self._histories[customer_id])  # copy: callers cannot mutate our state

    # -- similarity -------------------------------------------------------- #
    def similarity(self, customer_a: str, customer_b: str) -> float:
        self._require(customer_a)
        self._require(customer_b)
        return jaccard(self._histories[customer_a], self._histories[customer_b])

    def similar_customers(self, customer_id: str, k: int = 3) -> list[tuple[str, float]]:
        """Top-k other customers as (customer_id, score), best first.

        Excludes the customer, anyone with similarity 0 (sharing nothing is not
        'similar'), and anyone below min_similarity. Ties are broken by customer_id
        ascending, so results are deterministic.
        """
        self._require(customer_id)
        scored = []
        for other in self._histories:
            if other == customer_id:
                continue
            sim = self.similarity(customer_id, other)
            if sim > 0 and sim >= self.min_similarity:
                scored.append((other, sim))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    # -- candidates -------------------------------------------------------- #
    def candidate_products(self, customer_id: str, k: int = 3) -> dict[str, dict]:
        """Products bought by the k nearest neighbours but not by this customer.

        Evidence per product: supporters [(customer_id, similarity)], their summed
        similarity, and the product's popularity (distinct buyers and share of all
        customers).

        Fallback: if there are no neighbours, or the neighbours add nothing new,
        offer the most popular products the customer has not bought. These carry
        fallback=True and no supporters. A customer who already bought every product
        gets an empty dict.
        """
        owned = self.history(customer_id)
        neighbours = self.similar_customers(customer_id, k)

        candidates: dict[str, dict] = {}
        for neighbour_id, sim in neighbours:
            for pid in self._histories[neighbour_id] - owned:
                entry = candidates.setdefault(pid, {"supporters": [], "similarity_sum": 0.0})
                entry["supporters"].append((neighbour_id, sim))
                entry["similarity_sum"] += sim

        for pid, entry in candidates.items():
            entry["fallback"] = False
            self._add_popularity(pid, entry)

        if candidates:
            return candidates

        fallback: dict[str, dict] = {}
        for pid in self._products:
            if pid not in owned:
                entry = {"supporters": [], "similarity_sum": 0.0, "fallback": True}
                self._add_popularity(pid, entry)
                fallback[pid] = entry
        return fallback

    def _add_popularity(self, pid: str, entry: dict) -> None:
        entry["popularity"] = self._popularity[pid]
        entry["popularity_share"] = self._popularity[pid] / len(self._histories)

    # -- ranking ----------------------------------------------------------- #
    def score(self, evidence: dict) -> float:
        """score = sum of supporters' Jaccard similarities
                   + POPULARITY_WEIGHT * (share of all customers who bought it)

        The first term rewards products backed by MORE and CLOSER neighbours; the
        second is a small tie-breaking nudge towards products many people buy.
        Fallback candidates have no supporters, so they score on popularity alone
        and always rank below neighbour-backed products.
        """
        return evidence["similarity_sum"] + POPULARITY_WEIGHT * evidence["popularity_share"]

    def recommend(self, customer_id: str, n: int = 3) -> list[dict]:
        """Up to n ranked recommendations (best first) for one customer."""
        self._require(customer_id)
        ranked = []
        for pid, evidence in self.candidate_products(customer_id).items():
            ranked.append((self.score(evidence), pid, evidence))
        # Highest score, then more popular, then product_id: fully deterministic.
        ranked.sort(key=lambda t: (-t[0], -t[2]["popularity"], t[1]))

        results = []
        for rank, (score, pid, evidence) in enumerate(ranked[:n], start=1):
            info = self._products[pid]
            results.append({
                "customer_id": customer_id,
                "customer_name": self._customer_names[customer_id],
                "rank": rank,
                "product_id": pid,
                "product_name": info["name"],
                "category": info["category"],
                "score": score,
                "supporting_customers": sorted(c for c, _ in evidence["supporters"]),
                "fallback": evidence["fallback"],
                "popularity": evidence["popularity"],
            })
        return results

    def recommend_all(self, n: int = 3) -> list[dict]:
        """Recommendations for every customer. Customers with nothing to recommend
        get one placeholder row with blank product fields, so nobody disappears."""
        rows: list[dict] = []
        for cid in self.customers:
            recs = self.recommend(cid, n)
            if recs:
                rows.extend(recs)
            else:
                rows.append({
                    "customer_id": cid, "customer_name": self._customer_names[cid],
                    "rank": "", "product_id": "", "product_name": "", "category": "",
                    "score": "", "supporting_customers": [], "fallback": False,
                    "popularity": 0,
                })
        return rows

    # -- export ------------------------------------------------------------ #
    def export_csv(self, rows: list[dict], path: Path = OUTPUT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                out = {col: row.get(col, "") for col in CSV_COLUMNS}
                out["score"] = f"{row['score']:.3f}" if row.get("score") != "" else ""
                out["supporting_customers"] = ";".join(row.get("supporting_customers", []))
                writer.writerow(out)
