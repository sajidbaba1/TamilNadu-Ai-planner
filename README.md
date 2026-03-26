# Tamil Nadu Residential Floor Plan Generator

This project generates code-compliant residential floor plans for Tamil Nadu
plots using a hybrid pipeline of rule-based geometry, a SQLite knowledge
database, and trained ML models. Given a plot size, BHK count, road-facing
direction, and district, it produces a complete floor plan with rooms, walls,
doors, windows, material recommendations, Vastu scores, NBC compliance scores,
and SHAP-based explanations. Single-floor (G) and two-floor (G+1) layouts are
both supported.

---

## Requirements

- Python 3.13
- Packages listed in requirements.txt:
  - pandas, numpy, scikit-learn, xgboost, joblib, pyarrow
  - shapely, matplotlib, streamlit, ortools
  - tensorflow (for room_dimensions.h5 — install separately)
  - ezdxf (for DXF/CAD renderer — install separately)

---

## Setup

- Clone the repository
- Create and activate a virtual environment
- Run: pip install -r requirements.txt
- Install tensorflow: pip install tensorflow
- Install ezdxf: pip install ezdxf
- Place the three trained model files in the models\ folder:
  - constraint_scorer.pkl
  - room_dimensions.h5
  - shap_explainer.pkl
- Run the app: streamlit run app.py

---

## Project Structure

    layout_project/
      db/                   SQLite database, schema, build and validation scripts
      engine/               Core generation engine and public API wrapper
        engine.py           Band-based room placement, scoring, wall/door/window logic
        engine_api.py       generate_plan() — the only function app.py calls
      models/               Trained ML model artifacts (not committed to repo)
      renderer/             DXF/CAD renderer for floor plan drawings
      seeds/                11 CSV files encoding planning rules and constraints
      training_data/        50,000-sample synthetic parquet dataset
      outputs/              Generated plan images
      tests/                Integration and unit tests
      archive/              Legacy debug and development scripts
      generate_training_data.py   Generates synthetic training dataset
      retrain_models.py           Retrains XGBoost, Keras, and SHAP models
      test_integration.py         4-combination integration smoke test
      requirements.txt
      PRD.docx
      project.txt
      README.md

---

## How It Works

The pipeline has six phases:

1. Knowledge database — 11 CSV seed files (setbacks, climate zones, NBC codes,
   adjacency rules, materials, Baker principles, passage dimensions, plot
   configurations) are loaded into a SQLite database. All rule lookups during
   generation query this database.

2. Synthetic dataset — generate_training_data.py produces 50,000 floor plan
   samples with geometry features, violation flags, and validity labels. These
   are stored as a Parquet file and used to train the ML models.

3. Model training — three models are trained: an XGBoost classifier that scores
   plan validity (constraint_scorer.pkl), a Keras neural network that predicts
   room dimensions from plot parameters (room_dimensions.h5), and a SHAP
   TreeExplainer for interpretability (shap_explainer.pkl).

4. Engine — given plot parameters, the engine queries setbacks and climate data
   from the database, predicts room dimensions with the Keras model, places
   rooms in four horizontal bands (verandah, public, service, private), extracts
   a wall network, places doors and windows, builds a 119-feature vector, and
   scores the plan with the XGBoost model plus five deterministic rule checks.

5. Renderer — the renderer converts the FloorPlan object into a CAD drawing
   with colour-coded room fills, wall thickness, door swings, window openings,
   dimensions, and annotations.

6. App — a Streamlit interface accepts user inputs, calls generate_plan(),
   displays the scored plan with SHAP explanations, and provides PNG/DXF export.

---

## Supported Plot Sizes

Plot sizes follow Tamil Nadu residential plot bands as encoded in the
plot_configurations and tn_setbacks database tables:

- 60 sqm and below (EWS / small plots)
- 60 to 100 sqm
- 100 to 150 sqm
- 150 to 250 sqm (typical 9x12, 12x15 plots)
- 250 to 400 sqm (typical 15x20, 18x24 plots)
- 400 sqm and above (20x25 and larger)

BHK range supported: 1BHK to 4BHK. Floors: G (single) or G+1 (two-storey).

---

## Compliance

Plans are checked against the following codes and frameworks:

- NBC 2016 — National Building Code minimum room areas and widths
- TNCDBR 2019 — Tamil Nadu Combined Development and Building Rules setbacks
- CMDA / DTCP — Chennai and district-level planning authority setback rules
- Vastu Shastra — Room compass-zone placement scoring
- Baker principles — Laurie Baker passive cooling and cost-reduction guidelines
- Adjacency rules — Forbidden and required room-pair relationships from the
  adjacency_rules database table

---

## Sample Output

A generated plan for a 12x15m, 2BHK, North-facing plot in Coimbatore includes:

- Ground floor: master bedroom, bedroom 2, living, dining, kitchen, toilets,
  utility, verandah, staircase (G+1 plans)
- First floor (G+1): bedrooms 3 and 4, dry kitchen, toilets above ground
  wet rooms, staircase head, balcony
- Scores: validity (XGBoost), Vastu, NBC compliance, circulation, adjacency,
  and an overall weighted score
- Top 5 SHAP features explaining the validity score
- Climate-appropriate material recommendations for the district
- Applicable Baker passive-cooling principles with cost-saving percentages

---

## Known Limitations

The following limitations are documented in PRD Section 12:

1. The XGBoost constraint scorer was trained on synthetic data weighted toward
   larger plots (360 to 600 sqm). For plots below 200 sqm the score_valid
   output may be low even when the deterministic scores are high. This is a
   training distribution gap, not a geometry error.

2. All rooms are rectangular. Non-orthogonal walls, curved rooms, and
   split-level layouts are not supported in this version.

3. Structural grid alignment, utility routing (plumbing, electrical, sewage),
   and load-bearing wall placement are not modelled. The generated plan is an
   architectural space-planning output, not a structural drawing.

---

## Team

- Member 1
- Member 2
- Member 3
