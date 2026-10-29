# Customer Recommendation Engine — Mama Mboga Online

A "customers like you also bought" recommender built from first principles with
**sets, dictionaries and Jaccard similarity**. Standard library only — no ML libraries.
Every recommendation can be explained to a shop owner in one sentence:

> "Customers whose baskets look like yours bought this, and it's popular too."

## How to run

Requires Python 3.10+.

```bash
python main.py                 # runs end to end, writes output/recommendations.csv
pip install -r requirements.txt && python -m pytest -q    # optional: 16 tests
```

`main.py` prints the similarity matrix for the first five customers, the full
console view for two customers (C101 and C112), writes the CSV, and prints the run summary.
A copy of that console output is saved in `output/console_run.txt`.

## Project layout

```
recommender.py      read_purchases, jaccard, RecommendationEngine, UnknownCustomerError
main.py             end-to-end run + console output
data/purchases.csv  October 2026 orders
output/recommendations.csv
tests/test_recommender.py
```

## Approach

1. **Clean** the order export (`read_purchases`) into valid and rejected rows.
2. **Represent** each customer as a `set` of product IDs: `dict[customer_id -> set[product_id]]`.
   Product names/categories and popularity are also dicts. All are built **once** in
   `__init__`; later methods never re-scan the raw rows.
3. **Compare** customers with Jaccard similarity: `|A ∩ B| / |A ∪ B|`.
4. **Find neighbours**: the top-k most similar other customers (k = 3).
5. **Collect candidates**: products the neighbours bought that this customer has not,
   remembering which neighbours support each one.
6. **Rank** candidates with the scoring rule below and keep the top n (n = 3).

## Data-cleaning rules

The file has 63 rows. **60 are used, 3 are rejected.**

| Problem in the data | Rule | Result |
|---|---|---|
| `T9010` appears twice (identical row) | A `purchase_id` seen before is rejected; the first occurrence is kept | 1 row rejected (`duplicate_purchase_id`) |
| `T9020` has a blank product ID and name | A row with no `product_id` (or no `customer_id`) can't be placed in a basket | rejected (`missing_product_id`) |
| `T9033` has quantity `0` | Quantity must be a whole number > 0; nothing was actually bought | rejected (`invalid_quantity`) |
| `T9041` uses `c106` instead of `C106` | IDs are stripped and upper-cased, so it is the same customer | **kept**, merged into C106 |

Why these choices: counting the duplicate would not change a *set* but would inflate
popularity if popularity counted rows, so it is removed at the source. Merging `c106`
matters: without it Mwangi would look like two customers and one of them would have a
single-item history. Rejected rows carry a `reject_reason` field so they can be reviewed.

## Similarity and neighbours

* `jaccard(a, b)` uses the set operators `&` and `|`. **Two empty sets return 0.0** (no
  evidence they are alike) rather than dividing by zero.
* `similar_customers(customer, k)` excludes the customer, excludes anyone with similarity
  **0** (sharing nothing is not "similar"), and excludes anyone below `min_similarity`
  (default 0.0, configurable).
* **Ties** are broken by `customer_id` ascending, so results are identical on every run.
  Example: C108 and C120 both score 0.50 against C112, and C108 is listed first.

## Ranking rule

```
score = Σ similarity of each supporting neighbour
        + 0.25 × (share of all customers who bought the product)
```

* **Term 1 — similarity mass.** Each neighbour who bought the product "votes" with a weight
  equal to how similar they are. So a product backed by *more* and *closer* neighbours
  scores higher; one very similar neighbour can outweigh several distant ones.
* **Term 2 — popularity.** Products bought by many distinct customers get a small nudge
  (at most 0.25). It breaks near-ties towards safer picks without overpowering similarity.
  Popularity counts *distinct customers*, not rows or quantities, so one heavy buyer can't inflate it.
* Remaining ties are broken by popularity, then `product_id`.

### Worked example — customer C101 (Wanjiru)

Wanjiru bought `{P04 Blue Band, P08 Ketepa Tea}`. Her top-3 neighbours:

| Neighbour | Basket | Jaccard with C101 |
|---|---|---|
| C117 | P03, P04, P08, P10 | 2/4 = **0.50** |
| C109 | P03, P08 | 1/3 = **0.33** |
| C113 | P01, P02, P03, P04, P05 | 1/6 = **0.17** |

Two of the candidates (20 customers in total):

* **Pampers Size 3 (P10)** — supported by C117 only. 6 of 20 customers bought it.
  `score = 0.50 + 0.25 × (6/20) = 0.50 + 0.075 = 0.575`
* **Kienyeji Eggs (P01)** — supported by C113 only. 5 of 20 customers bought it.
  `score = 0.167 + 0.25 × (5/20) = 0.167 + 0.0625 = 0.229`

Pampers ranks above Eggs because its single supporter is three times closer to Wanjiru
(0.50 vs 0.17); popularity barely differs (6 vs 5 buyers). **Festive Bread (P03)** ranks
first overall (1.087) because *all three* neighbours bought it: 0.50 + 0.33 + 0.17 = 1.00,
plus a popularity bonus for 7 of 20 buyers.

## Edge cases

| Situation | Behaviour |
|---|---|
| **Customer with no neighbours** (shares nothing with anyone) | **Fallback:** recommend the most popular products they have not bought. Marked `fallback=True`, no supporting customers. Chosen because it is the standard, easy-to-explain cold-start answer. |
| Neighbours add nothing new (C107, C108, C120) | Same popularity fallback, since the customer would otherwise get nothing. |
| **Customer who bought everything** | Nothing to recommend; still gets **one row** in the CSV with blank product columns. |
| **Unknown customer ID** | Raises `UnknownCustomerError` (a `KeyError` subclass) with a readable message, e.g. `Unknown customer ID: 'C999'`. |
| Ties | Deterministic ordering for both neighbours and products (see above). |

## Results

```
Purchases read: 63 | used: 60 | rejected: 3
Customers: 20 | Products: 15 | Recommendations written: 52
```

All 20 customers appear in `output/recommendations.csv`. Fallback rows are the ones with
an empty `supporting_customers` column.

## Assumptions and limitations

* A customer's "history" is the **set** of products bought at least once in October;
  quantities and dates are ignored (a customer who bought Omo once and one who bought it
  five times look the same). Quantity-weighted similarity would be a natural next step.
* Customers who appear **only** in rejected rows would not exist in the engine. In this
  dataset every customer has at least one valid row.
* With only ~60 purchases, many baskets are 2–4 items, so similarities are coarse and
  the first five customers (C101–C105) share no products — their matrix is the identity.
* A customer may get fewer than `n` recommendations when their neighbours only know a few
  products they don't already have (e.g. C104 gets one). The popularity fallback is not
  used to "top up" lists, to keep every neighbour-based recommendation explainable.
* The popularity weight (0.25) and neighbourhood size (k = 3) are judgment calls on a tiny
  dataset, not tuned values.
