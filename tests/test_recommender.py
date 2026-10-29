import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommender import (  # noqa: E402
    RecommendationEngine, UnknownCustomerError, jaccard, read_purchases,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "purchases.csv"


def row(cid, pid, name="N", qty=1, pur=None):
    return {"purchase_id": pur or f"{cid}{pid}", "customer_id": cid, "customer_name": name,
            "product_id": pid, "product_name": f"Prod {pid}", "category": "Cat", "quantity": qty}


# ---- jaccard ---------------------------------------------------------------
def test_jaccard_basic():
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_both_empty_is_zero():
    assert jaccard(set(), set()) == 0.0


# ---- reading ---------------------------------------------------------------
def test_read_purchases_counts_and_reasons():
    valid, rejected = read_purchases(DATA)
    assert len(valid) == 60 and len(rejected) == 3
    reasons = {r["purchase_id"]: r["reject_reason"] for r in rejected}
    assert reasons == {"T9010": "duplicate_purchase_id", "T9020": "missing_product_id",
                       "T9033": "invalid_quantity"}


def test_lowercase_id_is_normalised():
    valid, _ = read_purchases(DATA)
    assert "c106" not in {r["customer_id"] for r in valid}
    assert sum(r["customer_id"] == "C106" for r in valid) == 3  # T9009, T9021, T9041


# ---- engine ----------------------------------------------------------------
@pytest.fixture
def engine():
    valid, _ = read_purchases(DATA)
    return RecommendationEngine(valid)


def test_similarity_properties(engine):
    for a in engine.customers:
        assert engine.similarity(a, a) == 1.0
        for b in engine.customers:
            s = engine.similarity(a, b)
            assert 0.0 <= s <= 1.0
            assert s == engine.similarity(b, a)


def test_similar_customers_excludes_self_and_is_sorted(engine):
    result = engine.similar_customers("C101", 5)
    assert "C101" not in [c for c, _ in result]
    sims = [s for _, s in result]
    assert sims == sorted(sims, reverse=True)
    assert all(s > 0 for s in sims)


def test_ties_are_broken_by_customer_id(engine):
    # C108 and C120 are both 0.50 from C112 -> C108 first
    top = engine.similar_customers("C112", 2)
    assert [c for c, _ in top] == ["C108", "C120"]


def test_recommendations_never_include_owned_products(engine):
    for cid in engine.customers:
        owned = engine.history(cid)
        assert not {r["product_id"] for r in engine.recommend(cid, 15)} & owned


def test_recommend_is_sorted_and_limited(engine):
    recs = engine.recommend("C101", 3)
    assert len(recs) <= 3
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in recs] == list(range(1, len(recs) + 1))


def test_unknown_customer_raises(engine):
    with pytest.raises(UnknownCustomerError):
        engine.recommend("C999")
    with pytest.raises(KeyError):
        engine.similar_customers("nope")


def test_history_returns_a_copy(engine):
    engine.history("C101").add("P99")
    assert "P99" not in engine.history("C101")


def test_new_customer_with_no_neighbours_gets_popular_fallback():
    engine = RecommendationEngine([
        row("C1", "P1"), row("C2", "P1"), row("C3", "P1"), row("C3", "P2"),
        row("C4", "P9"),  # shares nothing with anyone
    ])
    recs = engine.recommend("C4", 2)
    assert recs and all(r["fallback"] for r in recs)
    assert recs[0]["product_id"] == "P1"  # most popular (3 buyers)
    assert recs[0]["supporting_customers"] == []


def test_customer_who_bought_everything_gets_blank_row():
    engine = RecommendationEngine([row("C1", "P1"), row("C1", "P2"), row("C2", "P1")])
    assert engine.recommend("C1") == []
    placeholder = [r for r in engine.recommend_all() if r["customer_id"] == "C1"]
    assert len(placeholder) == 1 and placeholder[0]["product_id"] == ""


def test_min_similarity_filters_neighbours():
    purchases = [row("C1", "P1"), row("C1", "P2"), row("C2", "P1"), row("C2", "P3"),
                 row("C3", "P1"), row("C3", "P2"), row("C3", "P4")]
    loose = RecommendationEngine(purchases, min_similarity=0.0)
    strict = RecommendationEngine(purchases, min_similarity=0.7)
    assert len(loose.similar_customers("C1")) == 2
    assert strict.similar_customers("C1") == []


def test_popularity_counts_distinct_customers():
    engine = RecommendationEngine([row("C1", "P1", pur="a"), row("C1", "P1", pur="b"),
                                   row("C2", "P1")])
    assert engine.popularity("P1") == 2


def test_export_covers_every_customer(engine, tmp_path):
    out = tmp_path / "out" / "recs.csv"
    engine.export_csv(engine.recommend_all(3), out)
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["customer_id"] for r in rows} == set(engine.customers)
    assert list(rows[0]) == ["customer_id", "customer_name", "rank", "product_id",
                             "product_name", "category", "score", "supporting_customers"]
