"""Run the recommendation engine end to end."""
from recommender import DATA_PATH, OUTPUT_PATH, RecommendationEngine, read_purchases

DEMO_CUSTOMERS = ["C101", "C112"]


def print_similarity_matrix(engine: RecommendationEngine, count: int = 5) -> None:
    ids = engine.customers[:count]
    print("Similarity matrix (first five customers, Jaccard)")
    print("       " + "".join(f"{c:>7}" for c in ids))
    for a in ids:
        print(f"{a:>6} " + "".join(f"{engine.similarity(a, b):>7.2f}" for b in ids))
    print()


def print_customer(engine: RecommendationEngine, customer_id: str, n: int = 3) -> None:
    print(f"Customer {customer_id} ({engine.customer_name(customer_id)}) "
          f"bought: {sorted(engine.history(customer_id))}")
    print("Most similar customers:")
    neighbours = engine.similar_customers(customer_id, 3)
    if not neighbours:
        print("  (none - falling back to most popular products)")
    for other, sim in neighbours:
        print(f"  {other}  jaccard={sim:.2f}")
    print("Recommendations:")
    recs = engine.recommend(customer_id, n)
    if not recs:
        print("  (nothing to recommend)")
    for rec in recs:
        if rec["fallback"]:
            why = f"popular: {rec['popularity']} customers bought it"
        else:
            count = len(rec["supporting_customers"])
            why = f"{count} similar customer{'s' if count != 1 else ''} bought it"
        print(f"  {rec['rank']}. {rec['product_name']} ({rec['category']})  "
              f"score={rec['score']:.3f}  because {why}")
    print()


def main() -> None:
    valid, rejected = read_purchases(DATA_PATH)
    print(f"Rejected rows: {[(r['purchase_id'], r['reject_reason']) for r in rejected]}\n")

    engine = RecommendationEngine(valid)
    print_similarity_matrix(engine)
    for cid in DEMO_CUSTOMERS:
        print_customer(engine, cid)

    rows = engine.recommend_all(n=3)
    engine.export_csv(rows, OUTPUT_PATH)

    written = sum(1 for r in rows if r["product_id"])
    print(f"Purchases read: {len(valid) + len(rejected)} | used: {len(valid)} | "
          f"rejected: {len(rejected)}")
    print(f"Customers: {len(engine.customers)} | Products: {len(engine.products)} | "
          f"Recommendations written: {written}")


if __name__ == "__main__":
    main()
