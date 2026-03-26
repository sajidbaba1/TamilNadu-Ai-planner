# Architecture Decision Record — Tamil Nadu Floor Plan Engine

Read this before touching `engine/engine.py` or `engine/engine_api.py`.
Each entry answers: what the decision is, why it was made, what breaks if you change it.

---

## 1. FEATURE_COLS order (119 features, must match scorer exactly)

`build_feature_vector()` writes columns in a fixed order: base inputs, then
per-room geometry in `ROOM_UNIVERSE` order, then zone areas, then adjacency
flags. The XGBoost model (`constraint_scorer.pkl`) was trained with this exact
column sequence and stores it in `clf.feature_names_in_`. Column order is
everything for XGBoost — it ignores names at prediction time and reads by
index. `engine_api.py` guards this with a startup integrity check that compares
`feature_names_in_` against the vector built from a dummy plan; it raises
`ValueError` if either set or order diverges. If you add, remove, or reorder a
feature column without retraining, `clf.predict()` silently scores against the
wrong features, producing garbage validity scores with no error.

---

## 2. Sentinel value 0.0 for absent rooms (not -1.0)

When a room type is not present in a plan (e.g. no `pooja` in a 1BHK), all
five of its feature columns (`_w`, `_d`, `_area`, `_cx_pct`, `_cy_pct`) are
written as `0.0`. The training data (`generate_training_data.py`) also uses
`fillna(0.0)` for all numeric columns. XGBoost treats -1 as a real, negative
numeric value and will route those rows down different tree branches. Zero is
the only sentinel that is consistent between training and inference, and it
cleanly separates "absent" from "present but tiny". Change to -1 and the
model's validity scores for multi-BHK plans (which always have absent rooms)
will shift unpredictably.

---

## 3. Band placement order: pooja before kitchen before utility

Inside `_place_rooms()` Band 3, the placement sequence is:
`toilet_common` → `staircase` (G+1 only) → `utility` → `kitchen` → `store`
→ `pooja`. Pooja is sized first (before the kitchen loop) so its width can be
reserved at the east edge before `cluster_w` is split among utility/kitchen/
store. Kitchen must be placed before utility because `kitchen_x` and
`kitchen_w` are derived from the cluster boundaries; utility then fills the
gap west of kitchen. If you place utility first, the kitchen width calculation
reads a stale `util_w = 0` and the two rooms overlap. If you forget to reserve
pooja width before computing `_b3_avail_w`, pooja gets pushed outside the
buildable area and `try_add` silently drops it, causing a Vastu zone failure.

---

## 4. Kitchen y-anchor: must touch y_b2 (dining bottom wall), not y_b3 + utility_h

Kitchen is placed with `y = y_b3` and `depth = b3_h`, so its top edge is
exactly `y_b3 + b3_h = y_b2`. Dining sits in Band 2 with its bottom edge at
`y_b2`. This guarantees kitchen and dining share the wall at `y_b2`, satisfying
the REQUIRED adjacency rule from the database. An earlier version placed
kitchen at `y = y_b3 + utility_h`, creating a gap between kitchen's top and
dining's bottom — the adjacency check found no shared wall and failed the plan.
The `try_add` jitter loop uses `y_min = y_b3` (not lower) to prevent negative
dy shifts from re-introducing that gap. Do not change the y-anchor or the
y_min bound without verifying the `adj_kitchen_dining` feature is still 1.

---

## 5. EWS toilet NBC threshold: 1.2 sqm for net_area < 50 sqm (NBC 2016 EWS)

NBC 2016 Table 1 specifies a minimum toilet area of 2.5 sqm for standard
residential construction, but grants EWS (Economically Weaker Section) plots
an exemption down to roughly 1.2 sqm. For plots under 50 sqm (the EWS band),
the band heights leave so little room that `toilet_attached` and
`toilet_common` cannot physically reach 2.5 sqm without consuming the bedroom
entirely. `score_and_explain()` applies `nbc_min = 1.2` for toilets on these
plots so the NBC score does not penalise geometrically valid EWS plans. The
test suite uses the full NBC minimum without this exemption, so the toilet
depth formula (FIX 3) independently guarantees `area >= NBC * 0.88` for all
non-EWS plots. Remove the exemption and every 6×9 or 7×9 plan will score
NBC = 0 despite being compliant under the actual building code.

---

## 6. Staircase appears only when floors >= 2, placed at x = tc_w in Band 3

The staircase is a service-zone room, not a habitable one, so it belongs in
Band 3 alongside toilet_common. It is anchored at `x = tc_w` (immediately
east of the common toilet) because both rooms need plumbing-stack proximity and
because placing the staircase adjacent to the toilet allows the first-floor
`staircase_head` to land directly above the ground toilet — keeping wet-stack
columns aligned vertically as required by TNCDBR 2019. For G-only plans the
staircase is omitted entirely; adding one wastes Band 3 width and breaks the
`cluster_x` calculation (the staircase width is only subtracted when
`floors >= 2`). On the first floor, the staircase becomes `staircase_head` at
the exact same x, y position, satisfying the plumbing-stack alignment check
in `_score_first_floor`.

---

## 7. FIRST_FLOOR_SCORE_MAP: why dry_kitchen maps to kitchen, staircase_head to utility

The XGBoost scorer was trained only on ground-floor room types from
`ROOM_UNIVERSE`. First-floor plans contain three room types the model has never
seen: `dry_kitchen`, `staircase_head`, and `balcony`. Rather than retrain,
`engine_api.py` remaps these to their closest analogues before building the
feature vector: `dry_kitchen → kitchen` (same plumbing zone, same service band,
similar 2–3 sqm footprint), `staircase_head → utility` (same service band,
same 1–2.5 sqm footprint), `balcony → verandah` (same north-edge position,
same Band 1 geometry). The remap is applied only to the temporary placement
dict used for feature extraction; the `Room` objects passed to the renderer
keep their real room types so labels and colours are correct. Adding a new
first-floor room type without a remap entry will produce a zero-filled feature
column, which biases the validity score downward.

---

## 8. 15-attempt retry loop: why MAX_ATTEMPTS = 15 and threshold = 0.4

`_place_rooms()` uses a seeded RNG to jitter band proportions and room sizes.
Different seeds produce different layouts; some violate adjacency or Vastu
zones. The retry loop runs up to 15 seeds (seed, seed+1, …, seed+14), keeps
the best-scoring plan, and stops early the moment any plan scores above 0.4.
Fifteen attempts was chosen empirically: in testing across 200 random plots,
fewer than 1% required more than 12 seeds to find a valid plan, and 15 covers
all edge cases including small EWS plots where band heights are tightly
constrained. The 0.4 threshold is a "good enough" bar — the deterministic
rule checks (adjacency, NBC, Vastu) together contribute roughly 0.5 points;
a score above 0.4 means no major rule is violated and the XGBoost score is
also positive. Raising the threshold above 0.5 causes frequent exhaustion on
EWS plots; lowering it below 0.3 lets adjacency-failing plans through.
