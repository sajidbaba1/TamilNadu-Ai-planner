"""
engine_api.py — Public API wrapper for the floor-plan engine.

The only function app.py should ever call is ``generate_plan()``.
Engine internals (ModelLoader, _place_rooms, build_feature_vector, …)
are never exposed.
"""

import copy
import joblib
import os

from engine.engine import (
    generate as _engine_generate,
    ModelLoader,
    build_feature_vector,
    FloorPlan,
    Room,
    build_wall_network,
    place_doors,
    place_windows,
    score_and_explain,
    build_wall_geometry,
    get_window_scores,
    ROOM_LISTS,
    NBC_MIN_AREA,
    NBC_MIN_WIDTH,
    ROOM_UNIVERSE,
    _compass,
    CURRENT_FACING,
    FACING_MAP,
    CLIMATE_MAP,
    MODELS_DIR,
)


# ---------------------------------------------------------------------------
# TASK 3 — FEATURE_COLS integrity guard
# ---------------------------------------------------------------------------

_feature_alignment_checked = False


def check_feature_alignment():
    """Verify that build_feature_vector() output columns match the trained
    constraint_scorer model's expected feature names.

    Raises ``ValueError`` with a diff if they don't match exactly.
    Called once at engine startup (first ``generate_plan`` call).
    """
    global _feature_alignment_checked
    if _feature_alignment_checked:
        return

    scorer_path = os.path.join(MODELS_DIR, "constraint_scorer.pkl")
    try:
        clf = joblib.load(scorer_path)
    except FileNotFoundError:
        _feature_alignment_checked = True
        return
        
    model_features = list(getattr(clf, "feature_names_in_", []))
    if not model_features:
        _feature_alignment_checked = True
        return

    # Build a dummy FloorPlan to extract the columns build_feature_vector produces
    dummy_fp = FloorPlan(
        plot_w=12, plot_d=15, bhk=2, facing="N", district="test",
        net_w=10, net_d=12, setback_front=1.5, setback_rear=1.0,
        setback_side=1.0, facing_code=0, climate_code=0,
    )
    # Provide a minimal placement so the function runs
    dummy_fp.placement = {}
    df = build_feature_vector(dummy_fp, feature_cols=model_features)
    vector_cols = list(df.columns)

    model_set = set(model_features)
    vector_set = set(vector_cols)
    missing_in_vector = sorted(model_set - vector_set)
    extra_in_vector = sorted(vector_set - model_set)

    if missing_in_vector or extra_in_vector:
        parts = []
        if missing_in_vector:
            parts.append(f"  Missing from build_feature_vector: {missing_in_vector}")
        if extra_in_vector:
            parts.append(f"  Extra in build_feature_vector: {extra_in_vector}")
        raise ValueError(
            "FEATURE_COLS mismatch between constraint_scorer.pkl and "
            "build_feature_vector():\n" + "\n".join(parts)
        )

    # Also verify ordering
    if vector_cols != model_features:
        raise ValueError(
            "FEATURE_COLS ordering mismatch: build_feature_vector() columns "
            "do not match constraint_scorer.pkl feature_names_in_ order."
        )

    _feature_alignment_checked = True


# ---------------------------------------------------------------------------
# TASK 2 — G+1 first-floor generation
# ---------------------------------------------------------------------------

# Maps first-floor-only room types to their nearest ROOM_UNIVERSE equivalents
# for feature-vector / scoring purposes only.
# The Room object keeps its real room_type so the renderer sees the right label.
#
#   dry_kitchen    → kitchen    (same plumbing zone, same service band, similar footprint)
#   staircase_head → utility    (same service band, similar 1–2 m² footprint)
#   balcony        → verandah   (identical footprint and position; verandah is the column in the model)
#
# Room types that ARE already in ROOM_UNIVERSE (master_bedroom, bedroom_2..4,
# toilet_attached, toilet_common, pooja, store, dining, living, kitchen,
# utility, verandah) need no remapping.
FIRST_FLOOR_SCORE_MAP = {
    "dry_kitchen":    "kitchen",    # same plumbing zone, same service band
    "staircase_head": "utility",    # same service band, similar footprint
    "balcony":        "verandah",   # identical position; verandah is the trained column
    "staircase":      "utility",    # ground-floor staircase scores as utility (service band)
}


def _score_first_floor(first_fp: FloorPlan, clf, explainer) -> FloorPlan:
    """Score the first floor by remapping first-floor-only room types
    (dry_kitchen, staircase_head, balcony) to their nearest ROOM_UNIVERSE
    equivalents in the placement dict before calling score_and_explain.

    The Room objects (and thus the renderer) are not touched — only the
    temporary placement copy used for feature extraction is remapped.
    """
    # Build a remapped placement dict for the feature vector
    remapped = {}
    for rt, info in first_fp.placement.items():
        scorer_rt = FIRST_FLOOR_SCORE_MAP.get(rt, rt)
        # If multiple first-floor types map to the same scorer key,
        # keep the one with the larger area (better signal for the model).
        if scorer_rt in remapped:
            existing_area = remapped[scorer_rt]["w"] * remapped[scorer_rt]["d"]
            new_area = info["w"] * info["d"]
            if new_area <= existing_area:
                continue
        remapped[scorer_rt] = dict(info)  # copy so inner dicts can't alias back to original_placement

    # Swap placement, score, swap back
    original_placement = first_fp.placement
    first_fp.placement = remapped
    first_fp = score_and_explain(first_fp, clf, explainer)
    first_fp.placement = original_placement
    return first_fp


def _generate_first_floor(ground_fp: FloorPlan) -> FloorPlan:
    """Generate a first-floor plan based on the ground-floor layout.

    Rules (from PRD Section 6.2):
      - Same net_w, net_d, same setbacks
      - Verandah → balcony (same position, labelled differently)
      - Living/dining → additional bedroom(s)
        (1 bedroom for 2BHK G+1, 2 bedrooms for 3BHK G+1)
      - Kitchen → dry_kitchen / store_room
      - Staircase head room: 1.0m × 2.5m at same position as ground staircase
      - Wet rooms (toilets) directly above ground wet rooms (same x_abs, y_abs)
      - BHK label increments by 1 for first floor
    """
    gfp = ground_fp
    net_w = gfp.net_w
    net_d = gfp.net_d

    ground_rooms = {r.room_type: r for r in gfp.rooms}
    ground_bhk = gfp.bhk

    # First-floor BHK = ground BHK + 1
    first_bhk = ground_bhk + 1

    # Determine how many new bedrooms on first floor
    # 2BHK ground → 1 new bedroom on first floor
    # 3BHK ground → 2 new bedrooms on first floor
    new_bedroom_count = 1 if ground_bhk == 2 else 2

    first_rooms = []
    first_placement = {}

    # Track the next bedroom number (after ground-floor bedrooms)
    existing_bedrooms = [r for r in ground_rooms if r.startswith("bedroom_") or r == "master_bedroom"]
    max_bed_num = 1  # master_bedroom counts as 1
    for r in ground_rooms:
        if r.startswith("bedroom_"):
            try:
                num = int(r.split("_")[1])
                max_bed_num = max(max_bed_num, num)
            except (ValueError, IndexError):
                pass
    next_bed_num = max_bed_num + 1

    for rt, gr in ground_rooms.items():
        # ---- Verandah → Balcony ----
        if rt == "verandah":
            new_rt = "balcony"
            first_rooms.append(Room(
                room_type=new_rt,
                x=gr.x, y=gr.y, width=gr.width, depth=gr.depth,
                area=gr.area, cx_pct=gr.cx_pct, cy_pct=gr.cy_pct,
                compass=gr.compass,
            ))
            first_placement[new_rt] = {
                "x": gr.x, "y": gr.y, "w": gr.width, "d": gr.depth,
                "cx": gr.x + gr.width / 2.0, "cy": gr.y + gr.depth / 2.0,
            }
            continue

        # ---- Living / Dining → Additional bedrooms ----
        if rt in ("living", "dining"):
            if new_bedroom_count > 0:
                new_rt = f"bedroom_{next_bed_num}"
                first_rooms.append(Room(
                    room_type=new_rt,
                    x=gr.x, y=gr.y, width=gr.width, depth=gr.depth,
                    area=gr.area, cx_pct=gr.cx_pct, cy_pct=gr.cy_pct,
                    compass=gr.compass,
                ))
                first_placement[new_rt] = {
                    "x": gr.x, "y": gr.y, "w": gr.width, "d": gr.depth,
                    "cx": gr.x + gr.width / 2.0, "cy": gr.y + gr.depth / 2.0,
                }
                next_bed_num += 1
                new_bedroom_count -= 1
            else:
                # Extra public space: use next bedroom slot if one remains in
                # ROOM_UNIVERSE (bedroom_4 is the last), else fall back to "store".
                # "study" is NOT in ROOM_UNIVERSE and would crash the renderer.
                if ground_bhk >= 2 and next_bed_num <= 4:
                    new_rt = f"bedroom_{next_bed_num}"
                    next_bed_num += 1
                else:
                    new_rt = "store"
                first_rooms.append(Room(
                    room_type=new_rt,
                    x=gr.x, y=gr.y, width=gr.width, depth=gr.depth,
                    area=gr.area, cx_pct=gr.cx_pct, cy_pct=gr.cy_pct,
                    compass=gr.compass,
                ))
                first_placement[new_rt] = {
                    "x": gr.x, "y": gr.y, "w": gr.width, "d": gr.depth,
                    "cx": gr.x + gr.width / 2.0, "cy": gr.y + gr.depth / 2.0,
                }
            continue

        # ---- Kitchen → Dry kitchen / store room ----
        if rt == "kitchen":
            new_rt = "dry_kitchen"
            first_rooms.append(Room(
                room_type=new_rt,
                x=gr.x, y=gr.y, width=gr.width, depth=gr.depth,
                area=gr.area, cx_pct=gr.cx_pct, cy_pct=gr.cy_pct,
                compass=gr.compass,
            ))
            first_placement[new_rt] = {
                "x": gr.x, "y": gr.y, "w": gr.width, "d": gr.depth,
                "cx": gr.x + gr.width / 2.0, "cy": gr.y + gr.depth / 2.0,
            }
            continue

        # ---- Ground staircase → Staircase head room (same x, y, same 1.0m × 2.5m) ----
        # Placed at the exact same position so plumbing-stack alignment check passes.
        if rt == "staircase":
            first_rooms.append(Room(
                room_type="staircase_head",
                x=gr.x, y=gr.y, width=gr.width, depth=gr.depth,
                area=gr.area, cx_pct=gr.cx_pct, cy_pct=gr.cy_pct,
                compass=gr.compass,
            ))
            first_placement["staircase_head"] = {
                "x": gr.x, "y": gr.y, "w": gr.width, "d": gr.depth,
                "cx": gr.x + gr.width / 2.0, "cy": gr.y + gr.depth / 2.0,
            }
            continue

        # ---- Utility carries over unchanged ----
        if rt == "utility":
            first_rooms.append(Room(
                room_type="utility",
                x=gr.x, y=gr.y, width=gr.width, depth=gr.depth,
                area=gr.area, cx_pct=gr.cx_pct, cy_pct=gr.cy_pct,
                compass=gr.compass,
            ))
            first_placement["utility"] = {
                "x": gr.x, "y": gr.y, "w": gr.width, "d": gr.depth,
                "cx": gr.x + gr.width / 2.0, "cy": gr.y + gr.depth / 2.0,
            }
            continue

        # ---- Wet rooms (toilets) — same position above ground floor ----
        if rt in ("toilet_attached", "toilet_common"):
            first_rooms.append(Room(
                room_type=rt,
                x=gr.x, y=gr.y, width=gr.width, depth=gr.depth,
                area=gr.area, cx_pct=gr.cx_pct, cy_pct=gr.cy_pct,
                compass=gr.compass,
            ))
            first_placement[rt] = {
                "x": gr.x, "y": gr.y, "w": gr.width, "d": gr.depth,
                "cx": gr.x + gr.width / 2.0, "cy": gr.y + gr.depth / 2.0,
            }
            continue

        # ---- Bedrooms carry over (same position) ----
        if rt.startswith("bedroom_") or rt == "master_bedroom":
            first_rooms.append(Room(
                room_type=rt,
                x=gr.x, y=gr.y, width=gr.width, depth=gr.depth,
                area=gr.area, cx_pct=gr.cx_pct, cy_pct=gr.cy_pct,
                compass=gr.compass,
            ))
            first_placement[rt] = {
                "x": gr.x, "y": gr.y, "w": gr.width, "d": gr.depth,
                "cx": gr.x + gr.width / 2.0, "cy": gr.y + gr.depth / 2.0,
            }
            continue

        # ---- Pooja / store — carry over as-is ----
        if rt in ("pooja", "store"):
            first_rooms.append(Room(
                room_type=rt,
                x=gr.x, y=gr.y, width=gr.width, depth=gr.depth,
                area=gr.area, cx_pct=gr.cx_pct, cy_pct=gr.cy_pct,
                compass=gr.compass,
            ))
            first_placement[rt] = {
                "x": gr.x, "y": gr.y, "w": gr.width, "d": gr.depth,
                "cx": gr.x + gr.width / 2.0, "cy": gr.y + gr.depth / 2.0,
            }
            continue

    # Build the first-floor FloorPlan object
    first_fp = FloorPlan(
        plot_w=gfp.plot_w,
        plot_d=gfp.plot_d,
        bhk=first_bhk,
        facing=gfp.facing,
        district=gfp.district,
        net_w=net_w,
        net_d=net_d,
        setback_front=gfp.setback_front,
        setback_rear=gfp.setback_rear,
        setback_side=gfp.setback_side,
        climate_zone=gfp.climate_zone,
        facing_code=gfp.facing_code,
        climate_code=gfp.climate_code,
        materials=gfp.materials,
        baker_principles=gfp.baker_principles,
        seed=gfp.seed,
    )
    first_fp.rooms = first_rooms
    first_fp.placement = first_placement
    first_fp.band_b4_h = gfp.band_b4_h
    first_fp.band_y_b3 = gfp.band_y_b3
    first_fp.band_y_b2 = gfp.band_y_b2

    # Build walls, doors, windows for first floor
    first_fp.walls = build_wall_network(first_fp.rooms, net_w, net_d)
    window_scores = get_window_scores(gfp.district)

    import engine.engine as _eng
    saved_facing = _eng.CURRENT_FACING
    _eng.CURRENT_FACING = gfp.facing
    first_fp.doors = place_doors(first_fp.rooms, first_fp.walls, first_bhk)
    first_fp.windows = place_windows(
        first_fp.rooms, first_fp.walls, window_scores, gfp.facing
    )
    _eng.CURRENT_FACING = saved_facing

    # Score first floor — uses remapped placement so dry_kitchen/staircase_head/
    # balcony map to their ROOM_UNIVERSE equivalents (see FIRST_FLOOR_SCORE_MAP).
    clf, _, explainer = ModelLoader.get()
    first_fp = _score_first_floor(first_fp, clf, explainer)
    first_fp.wall_geometry = build_wall_geometry(first_fp)

    return first_fp


# ---------------------------------------------------------------------------
# TASK 1 — Public API
# ---------------------------------------------------------------------------

def generate_plan(
    plot_w,
    plot_d,
    bhk,
    facing,
    district,
    floors=1,
    vastu=True,
    baker=True,
    seed=None,
):
    """Generate a complete floor plan (G or G+1).

    Parameters
    ----------
    plot_w, plot_d : float
        Plot width and depth in metres.
    bhk : int
        Number of bedrooms (1–4).
    facing : str
        Road-facing direction — ``'N'``, ``'S'``, ``'E'``, or ``'W'``.
    district : str
        District name (used for setbacks, climate, materials lookup).
    floors : int, optional
        1 for ground only, 2 for G+1.  Default 1.
    vastu : bool, optional
        Reserved for future vastu-toggle.  Currently always applied.
    baker : bool, optional
        Reserved for future baker-toggle.  Currently always applied.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``"ground"``   – :class:`FloorPlan` for the ground floor.
        ``"first"``    – :class:`FloorPlan` for the first floor, or *None*
                         when ``floors == 1``.
        ``"scores"``   – dict of all 7 scores (ground floor).
        ``"metadata"`` – district, climate_zone, soil_type, materials,
                         baker_principles, seed_used.
    """
    # Run feature-alignment guard once at startup
    check_feature_alignment()

    # Generate ground floor via the engine
    params = {
        "plot_w": float(plot_w),
        "plot_d": float(plot_d),
        "bhk": int(bhk),
        "facing": str(facing).upper(),
        "district": str(district),
    }
    if seed is not None:
        params["seed"] = int(seed)
    params["floors"] = int(floors)

    ground_fp = _engine_generate(params)

    # Generate first floor if requested.
    # NOTE: _generate_first_floor must run while ground_fp.rooms is still a
    # List[Room] — it iterates the list to build ground_rooms dict internally.
    first_fp = None
    if floors >= 2:
        first_fp = _generate_first_floor(ground_fp)
        
    second_fp = None
    if floors >= 3:
        second_fp = copy.deepcopy(first_fp)

    # Convert fp.rooms from List[Room] → dict[room_type, Room] on both floors.
    # This is the public API contract: callers use rooms.keys() / rooms.get().
    # Engine internals (renderer, wall builder, scorer) have already finished
    # by this point and operated on the list form.
    ground_fp.rooms = {r.room_type: r for r in ground_fp.rooms}
    if first_fp is not None:
        first_fp.rooms = {r.room_type: r for r in first_fp.rooms}
    if second_fp is not None:
        second_fp.rooms = {r.room_type: r for r in second_fp.rooms}

    # Collect all 7 scores from the ground floor
    scores = {
        "score_valid": ground_fp.score_valid,
        "score_vastu": ground_fp.score_vastu,
        "score_nbc": ground_fp.score_nbc,
        "score_circulation": ground_fp.score_circulation,
        "score_adjacency": ground_fp.score_adjacency,
        "score_overall": ground_fp.score_overall,
    }

    # Add first-floor scores if present
    if first_fp is not None:
        scores["first_score_valid"] = first_fp.score_valid
        scores["first_score_vastu"] = first_fp.score_vastu
        scores["first_score_nbc"] = first_fp.score_nbc
        scores["first_score_circulation"] = first_fp.score_circulation
        scores["first_score_adjacency"] = first_fp.score_adjacency
        scores["first_score_overall"] = first_fp.score_overall

    metadata = {
        "district": ground_fp.district,
        "climate_zone": ground_fp.climate_zone,
        "soil_type": "alluvial",  # default; extend from DB if available
        "materials": ground_fp.materials,
        "baker_principles": ground_fp.baker_principles,
        "seed_used": ground_fp.seed,
    }

    return {
        "ground": ground_fp,
        "first": first_fp,
        "second": second_fp,
        "scores": scores,
        "metadata": metadata,
    }
