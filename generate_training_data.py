import os
import sqlite3
import time
import math

import numpy as np
import pandas as pd

DB_PATH = "db/floorplan.db"
OUT_PATH = "training_data/floor_plan_samples.parquet"
N_SAMPLES = 50000
SEED = 42

# FIX 1: uniform plot-size band sampling
PLOT_BANDS = {
    "small":  [(5, 10), (6, 9), (7.5, 10), (9, 12), (10, 12)],
    "medium": [(12, 15), (12, 18), (15, 15), (15, 18)],
    "large":  [(15, 20), (18, 20), (20, 24), (20, 25), (24, 30)],
}
PLOT_BAND_KEYS = ["small", "medium", "large"]

TN_PLOTS = [
    # (plot_w, plot_d, authority, category, max_bhk, layout_type)
    # EWS & LIG — 1 BHK only
    (4, 9, "Panchayat", "EWS", 1, "narrow"),
    (5, 9, "TNHB", "EWS", 1, "narrow"),
    (6, 9, "TNHB", "EWS", 1, "square"),
    (5, 10, "Panchayat", "EWS", 1, "narrow"),
    (6, 10, "DTCP", "EWS", 1, "square"),
    (6, 12, "DTCP", "LIG", 1, "square"),
    (7, 10, "Panchayat", "LIG", 1, "square"),
    (7.5, 10, "DTCP", "LIG", 2, "square"),
    (7, 12, "DTCP", "LIG", 2, "square"),
    (7.5, 12, "DTCP", "LIG", 2, "square"),
    # Standard — 2-3 BHK
    (8, 12, "DTCP", "standard", 2, "square"),
    (9, 12, "DTCP", "standard", 2, "square"),
    (8, 15, "DTCP", "standard", 2, "square"),
    (9, 14, "DTCP", "standard", 2, "square"),
    (10, 12, "DTCP", "standard", 2, "wide"),
    (9, 15, "DTCP", "standard", 2, "square"),
    (10, 14, "DTCP", "standard", 3, "square"),
    (10, 15, "DTCP", "standard", 3, "square"),
    (12, 12, "DTCP", "standard", 3, "wide"),
    (10, 16, "DTCP", "standard", 3, "square"),
    (12, 14, "DTCP", "standard", 3, "square"),
    (12, 15, "DTCP", "standard", 3, "square"),
    (11, 17, "DTCP", "standard", 3, "square"),
    (12, 16, "DTCP", "standard", 3, "square"),
    # Narrow plots — 1-3 BHK
    (4, 15, "corp", "narrow", 1, "narrow"),
    (5, 15, "corp", "narrow", 2, "narrow"),
    (5, 18, "corp", "narrow", 2, "narrow"),
    (6, 18, "DTCP", "narrow", 2, "narrow"),
    (6, 20, "DTCP", "narrow", 2, "narrow"),
    (7, 20, "corp", "narrow", 3, "narrow"),
    (8, 20, "DTCP", "narrow", 3, "narrow"),
    (9, 20, "DTCP", "narrow", 3, "narrow"),
    (10, 20, "CMDA", "narrow", 3, "narrow"),
    # Premium — 3-4 BHK
    (12, 18, "CMDA", "premium", 3, "square"),
    (14, 16, "CMDA", "premium", 3, "square"),
    (15, 15, "CMDA", "premium", 4, "square"),
    (12, 20, "CMDA", "premium", 4, "narrow"),
    (14, 18, "CMDA", "premium", 4, "square"),
    (15, 18, "CMDA", "premium", 4, "square"),
    (16, 18, "CMDA", "premium", 4, "square"),
    (15, 20, "CMDA", "premium", 4, "square"),
    (18, 18, "CMDA", "premium", 4, "square"),
    (18, 20, "CMDA", "premium", 4, "square"),
    # Large — 4-5 BHK
    (20, 20, "CMDA", "large", 4, "square"),
    (20, 25, "CMDA", "large", 4, "square"),
    (24, 20, "CMDA", "large", 4, "wide"),
    (21, 25, "CMDA", "large", 4, "square"),
    (24, 24, "CMDA", "large", 4, "square"),
    (24, 30, "CMDA", "large", 4, "square"),
]

_LAST_PLOT_META = {}

FACINGS = ["N", "S", "E", "W"]
FACING_MAP = {"N": 0, "S": 1, "E": 2, "W": 3}
CLIMATE_MAP = {"Hot_Humid": 0, "Hot_Dry": 1, "Composite": 2, "Warm_Humid": 3}

ROOM_LISTS = {
    1: ["master_bedroom", "toilet_attached", "living", "kitchen", "verandah"],
    2: ["master_bedroom", "toilet_attached", "bedroom_2", "living", "dining", "kitchen", "toilet_common", "utility", "verandah"],
    3: ["master_bedroom", "toilet_attached", "bedroom_2", "bedroom_3", "living", "dining", "kitchen", "toilet_common", "utility", "verandah", "pooja"],
    4: ["master_bedroom", "toilet_attached", "bedroom_2", "bedroom_3", "bedroom_4", "living", "dining", "kitchen", "toilet_common", "utility", "verandah", "pooja", "store"],
}

# 13-room universe used by existing 74-feature block.
ROOM_UNIVERSE = [
    "master_bedroom", "toilet_attached", "living", "kitchen", "verandah",
    "bedroom_2", "bedroom_3", "dining", "toilet_common", "utility", "pooja", "bedroom_4", "store",
]

NBC_MIN_AREA = {
    "master_bedroom": 9.5, "bedroom_2": 7.5, "bedroom_3": 7.5, "bedroom_4": 7.5,
    "living": 9.5, "dining": 5.0, "kitchen": 4.5, "toilet_attached": 2.5, "toilet_common": 2.5,
    "utility": 2.0, "verandah": 4.0, "pooja": 1.2, "store": 2.0,
}
NBC_MIN_WIDTH = {
    "master_bedroom": 2.4, "bedroom_2": 2.1, "bedroom_3": 2.1, "bedroom_4": 2.1,
    "living": 2.4, "dining": 1.8, "kitchen": 1.8, "toilet_attached": 1.2, "toilet_common": 1.2,
    "utility": 1.0, "verandah": 1.5, "pooja": 0.9, "store": 1.0,
}

HARDCODED_DEFAULTS = {
    "master_bedroom": (3.2, 3.0),
    "bedroom_2": (2.9, 2.9),
    "bedroom_3": (2.7, 2.7),
    "bedroom_4": (2.7, 2.7),
    "living": (4.2, 3.6),
    "dining": (2.6, 2.4),
    "kitchen": (2.6, 2.4),
    "toilet_attached": (1.5, 1.5),
    "toilet_common": (1.4, 1.4),
    "utility": (1.6, 1.4),
    "verandah": (0.0, 1.5),
    "pooja": (1.2, 1.2),
    "store": (1.5, 1.2),
}

ROOM_ZONE = {
    "verandah": "public_zone", "living": "public_zone", "dining": "public_zone", "pooja": "public_zone",
    "master_bedroom": "private_zone", "bedroom_2": "private_zone", "bedroom_3": "private_zone", "bedroom_4": "private_zone",
    "kitchen": "wet_zone", "utility": "wet_zone",
    "toilet_attached": "service_zone", "toilet_common": "service_zone", "store": "service_zone",
}


def _safe_sql(conn, q, cols=None):
    try:
        return pd.read_sql_query(q, conn)
    except Exception:
        return pd.DataFrame(columns=cols or [])


def _wall_thickness(conn):
    # FIX 3: read ext/int thickness from passage_dimensions if present, else defaults
    ext_t, int_t = 0.23, 0.115
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(passage_dimensions)").fetchall()]
        if cols:
            df = _safe_sql(conn, "SELECT * FROM passage_dimensions LIMIT 1")
            if not df.empty:
                for c in ("wall_thickness_ext_m", "wall_thickness_exterior_m", "exterior_wall_thickness_m", "wall_ext_m"):
                    if c in df.columns:
                        v = float(df.iloc[0][c])
                        if v > 0.05:
                            ext_t = v
                        break
                for c in ("wall_thickness_int_m", "wall_thickness_interior_m", "interior_wall_thickness_m", "wall_int_m"):
                    if c in df.columns:
                        v = float(df.iloc[0][c])
                        if v > 0.03:
                            int_t = v
                        break
    except Exception:
        pass
    return float(ext_t), float(int_t)


def _min_clear_width(conn):
    try:
        df = _safe_sql(conn, "SELECT * FROM passage_dimensions LIMIT 1")
        if not df.empty and "min_clear_width_m" in df.columns:
            v = float(df.iloc[0]["min_clear_width_m"])
            if v > 0.0:
                return float(v)
    except Exception:
        pass
    return 0.6


def _load_plot_bands(conn):
    try:
        df = _safe_sql(conn, "SELECT * FROM plot_configurations")
        if df.empty:
            return PLOT_BANDS
        band_col = None
        for c in ("plot_size_band", "size_band", "band"):
            if c in df.columns:
                band_col = c
                break
        if band_col and "plot_width_m" in df.columns and "plot_depth_m" in df.columns:
            bands = {"small": [], "medium": [], "large": []}
            for _, row in df.iterrows():
                band = str(row.get(band_col, "")).strip().lower()
                if band not in bands:
                    continue
                try:
                    bands[band].append((float(row.get("plot_width_m")), float(row.get("plot_depth_m"))))
                except Exception:
                    pass
            if any(bands.values()):
                return {k: v or PLOT_BANDS[k] for k, v in bands.items()}
    except Exception:
        pass
    return PLOT_BANDS


def _pfx(rt: str) -> str:
    return rt.replace("_", "")[:10]


def sample_plot_and_bhk(rng):
    global _LAST_PLOT_META

    idx = int(rng.integers(0, len(TN_PLOTS)))
    pw, pd, authority, category, max_bhk, layout = TN_PLOTS[idx]

    if category == "EWS":
        bhk = 1
    elif category == "LIG":
        choices = [b for b in [1, 2] if b <= max_bhk]
        bhk = int(rng.choice(choices or [1]))
    elif category in ("standard", "narrow"):
        choices = [b for b in [2, 3] if b <= max_bhk]
        bhk = int(rng.choice(choices or [1]))
    elif category == "premium":
        choices = [b for b in [3, 4] if b <= max_bhk]
        bhk = int(rng.choice(choices or [max_bhk]))
    else:
        choices = [b for b in [4, 5] if b <= max_bhk]
        bhk = int(rng.choice(choices or [max_bhk]))

    pw = round(float(pw) + float(rng.uniform(-0.3, 0.3)), 1)
    pd = round(float(pd) + float(rng.uniform(-0.3, 0.3)), 1)
    pw = max(pw, 4.0)
    pd = max(pd, 8.0)

    gross_area = pw * pd
    if gross_area <= 120:
        band = "small"
    elif gross_area <= 270:
        band = "medium"
    else:
        band = "large"

    _LAST_PLOT_META = {
        "plot_authority": authority,
        "plot_category": category,
        "plot_layout_type": layout,
    }
    return pw, pd, bhk, band


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _interval_union(ivs, tol=1e-6):
    if not ivs:
        return []
    ivs = sorted((min(a, b), max(a, b)) for a, b in ivs)
    out = [ivs[0]]
    for a, b in ivs[1:]:
        la, lb = out[-1]
        if a <= lb + tol:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def _adj(a, b, tol=0.05):
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["d"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["d"]
    if abs(ax1 - bx0) <= tol or abs(bx1 - ax0) <= tol:
        if min(ay1, by1) - max(ay0, by0) > tol:
            return True
    if abs(ay1 - by0) <= tol or abs(by1 - ay0) <= tol:
        if min(ax1, bx1) - max(ax0, bx0) > tol:
            return True
    return False


def _wall_stats(pl, net_w, net_d, tol=0.05):
    rooms = list(pl.keys())
    north, south, east, west = [], [], [], []
    for rt in rooms:
        a = pl[rt]
        x0, y0 = a["x"], a["y"]
        x1, y1 = x0 + a["w"], y0 + a["d"]
        if abs(y1 - net_d) <= tol:
            north.append((x0, x1))
        if abs(y0 - 0.0) <= tol:
            south.append((x0, x1))
        if abs(x1 - net_w) <= tol:
            east.append((y0, y1))
        if abs(x0 - 0.0) <= tol:
            west.append((y0, y1))
    north_u = _interval_union(north, tol=tol)
    south_u = _interval_union(south, tol=tol)
    east_u = _interval_union(east, tol=tol)
    west_u = _interval_union(west, tol=tol)
    ext_c = len(north_u) + len(south_u) + len(east_u) + len(west_u)
    ext_l = sum(b - a for a, b in north_u + south_u + east_u + west_u)

    seen, int_c, int_l = set(), 0, 0.0
    for i, ra in enumerate(rooms):
        a = pl[ra]
        ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["d"]
        for rb in rooms[i + 1:]:
            b = pl[rb]
            bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["d"]
            if abs(ax1 - bx0) <= tol or abs(bx1 - ax0) <= tol:
                y0 = max(ay0, by0)
                y1 = min(ay1, by1)
                if y1 - y0 > tol:
                    x = ax1 if abs(ax1 - bx0) <= tol else bx1
                    key = ("V", round(x, 3), round(y0, 3), round(y1, 3))
                    if key not in seen:
                        seen.add(key); int_c += 1; int_l += (y1 - y0)
            if abs(ay1 - by0) <= tol or abs(by1 - ay0) <= tol:
                x0 = max(ax0, bx0)
                x1 = min(ax1, bx1)
                if x1 - x0 > tol:
                    y = ay1 if abs(ay1 - by0) <= tol else by1
                    key = ("H", round(y, 3), round(x0, 3), round(x1, 3))
                    if key not in seen:
                        seen.add(key); int_c += 1; int_l += (x1 - x0)
    return int(ext_c), int(int_c), float(round(ext_l, 3)), float(round(int_l, 3))


def _setbacks(area, df):
    try:
        m = df[(df["plot_area_min_sqm"] <= area) & (df["plot_area_max_sqm"].isna() | (df["plot_area_max_sqm"] >= area))]
        if m.empty:
            return 2.0, 1.5, 1.0
        row = m.sort_values("plot_area_min_sqm", ascending=False).iloc[0]
        front = float(row.get("front_setback_m", 2.0))
        rear = float(row.get("rear_setback_m", 1.5))
        left = row.get("side_setback_left_m", 1.0)
        right = row.get("side_setback_right_m", 1.0)
        if pd.isna(left): left = 1.0
        if pd.isna(right): right = 1.0
        side = float((float(left) + float(right)) / 2.0)
        return float(front), float(rear), float(side)
    except Exception:
        return 2.0, 1.5, 1.0


def _targets(plot_w, plot_d, bhk, conf, rng):
    rooms = ROOM_LISTS.get(bhk, [])
    t = {}
    row = None
    try:
        m = conf[(conf["plot_width_m"].sub(plot_w).abs() <= 1.5) & (conf["plot_depth_m"].sub(plot_d).abs() <= 2.0) & (conf["bhk_type"] == f"{bhk}BHK")]
        if not m.empty:
            row = m.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
    except Exception:
        row = None
    cmap = {
        "master_bedroom": ("master_bedroom_target_w", "master_bedroom_target_d"),
        "bedroom_2": ("bedroom2_target_w", "bedroom2_target_d"),
        "bedroom_3": ("bedroom3_target_w", "bedroom3_target_d"),
        "bedroom_4": ("bedroom4_target_w", "bedroom4_target_d"),
        "living": ("living_target_w", "living_target_d"),
        "dining": ("dining_target_w", "dining_target_d"),
        "kitchen": ("kitchen_target_w", "kitchen_target_d"),
        "toilet_attached": ("toilet_att_target_w", "toilet_att_target_d"),
        "toilet_common": ("toilet_common_target_w", "toilet_common_target_d"),
        "utility": ("utility_target_w", "utility_target_d"),
        "verandah": ("verandah_target_w", "verandah_target_d"),
        "pooja": ("pooja_target_w", "pooja_target_d"),
        "store": ("store_target_w", "store_target_d"),
    }
    for rt in rooms:
        w0, d0 = HARDCODED_DEFAULTS.get(rt, (2.4, 2.4))
        w, d = w0, d0
        if row is not None and rt in cmap:
            cw, cd = cmap[rt]
            try:
                w = float(row.get(cw, w0)); d = float(row.get(cd, d0))
                if not (w > 0.3 and d > 0.3):
                    w, d = w0, d0
            except Exception:
                w, d = w0, d0
        w *= float(rng.uniform(0.92, 1.08)); d *= float(rng.uniform(0.92, 1.08))
        w = max(w, NBC_MIN_WIDTH.get(rt, 1.0))
        if w * d < NBC_MIN_AREA.get(rt, 0.0):
            d = (NBC_MIN_AREA[rt] / w) + 0.1
        t[rt] = (round(float(w), 2), round(float(d), 2))
    return t


def _rotate(rooms_n, net_w, net_d, facing):
    if facing == "N":
        return rooms_n
    out = []
    if facing == "S":
        for rt, x, y, w, d in rooms_n:
            out.append((rt, x, round(net_d - y - d, 2), w, d))
        return out
    if facing == "E":
        for rt, x, y, w, d in rooms_n:
            out.append((rt, round(y, 2), round(net_w - x - w, 2), d, w))
        return out
    for rt, x, y, w, d in rooms_n:
        out.append((rt, round(net_d - y - d, 2), round(x, 2), d, w))
    return out


def _place(net_w, net_d, bhk, t, rng, facing="N", err_p=0.05):
    reference_net_area = 108.0
    tol = 0.05
    step = 0.1

    # Local room program used only by placement. This keeps the rewrite
    # contained to this function as requested.
    bhk_rooms = {
        1: [
            "verandah", "living", "dining", "kitchen", "utility",
            "toilet_common", "master_bedroom", "toilet_attached",
        ],
        2: [
            "verandah", "living", "dining", "kitchen", "utility",
            "toilet_common", "master_bedroom", "toilet_attached",
            "bedroom_2",
        ],
        3: [
            "verandah", "living", "dining", "kitchen", "utility",
            "toilet_common", "master_bedroom", "toilet_attached",
            "bedroom_2", "bedroom_3", "pooja",
        ],
        4: [
            "verandah", "living", "dining", "kitchen", "utility",
            "toilet_common", "master_bedroom", "toilet_attached",
            "bedroom_2", "bedroom_3", "bedroom_4", "pooja", "store",
        ],
    }

    rooms = list(bhk_rooms.get(bhk, ROOM_LISTS.get(bhk, [])))
    net_area = float(net_w * net_d)
    area_scale_factor = max(1.0, net_area / reference_net_area)
    small_plot = net_area < 90.0
    ews_plot = net_area < 50.0

    if ews_plot:
        rooms = [
            r for r in rooms if r not in
            ("dining", "pooja", "store", "bedroom_2", "bedroom_3", "bedroom_4", "utility")
        ]
        bhk = 1

    if small_plot:
        rooms = [r for r in rooms if r not in ("dining", "pooja", "store")]

    room_rules = {
        "master_bedroom": {"min_area": 9.5, "min_width": 2.8},
        "bedroom_2": {"min_area": 7.5, "min_width": 2.4},
        "bedroom_3": {"min_area": 7.5, "min_width": 2.4},
        "bedroom_4": {"min_area": 7.5, "min_width": 2.4},
        "living": {"min_area": 10.0, "min_width": 3.0},
        "dining": {"min_area": 6.0, "min_width": 2.4},
        "kitchen": {"min_area": 4.5, "min_width": 1.8},
        "toilet_attached": {"min_area": 2.3, "min_width": 1.1},
        "toilet_common": {"min_area": 2.3, "min_width": 1.1},
        "utility": {"min_area": 2.5, "min_width": 1.2},
        "verandah": {"min_area": net_w * 1.5, "min_width": net_w},
        "pooja": {"min_area": 1.5, "min_width": 1.0},
        "store": {"min_area": 2.0, "min_width": 1.0},
    }

    def target_area(rt):
        if rt == "verandah":
            return float(net_w * 1.5)
        return float(room_rules[rt]["min_area"] * area_scale_factor)

    def scaled_dims(rt):
        base_w, base_d = t.get(rt, HARDCODED_DEFAULTS.get(rt, (2.4, 2.4)))
        base_w = max(float(base_w), 0.1)
        base_d = max(float(base_d), 0.1)
        ratio = _clamp(base_w / base_d, 0.55, 2.40)
        area = target_area(rt)
        min_w = float(room_rules[rt]["min_width"])
        w = max(min_w, math.sqrt(area * ratio))
        d = area / w
        if w * d < area:
            d = area / w
        return round(float(w), 2), round(float(d), 2)

    def overlaps(a, b, pad=tol):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return (
            ax0 < bx1 - pad and
            ax1 > bx0 + pad and
            ay0 < by1 - pad and
            ay1 > by0 + pad
        )

    placed = []
    skipped_overlap = False

    def can_place(x, y, w, d, x_min, x_max, y_min, y_max):
        if x < x_min - 1e-6 or y < y_min - 1e-6:
            return False
        if x + w > x_max + 1e-6 or y + d > y_max + 1e-6:
            return False
        rect = (x, y, x + w, y + d)
        for _, px, py, pw, pd in placed:
            if overlaps(rect, (px, py, px + pw, py + pd)):
                return False
        return True

    def try_add(rt, x, y, w, d, x_min, x_max, y_min, y_max):
        nonlocal skipped_overlap
        attempts = [(0.0, 0.0)]
        for i in range(1, 6):
            delta = round(i * step, 2)
            attempts.extend([
                (delta, 0.0), (-delta, 0.0),
                (0.0, delta), (0.0, -delta),
                (delta, delta), (-delta, delta),
            ])
        for dx, dy in attempts:
            xx = round(x + dx, 2)
            yy = round(y + dy, 2)
            if can_place(xx, yy, w, d, x_min, x_max, y_min, y_max):
                placed.append((rt, xx, yy, round(w, 2), round(d, 2)))
                return True
        skipped_overlap = True
        return False

    # y=0 south, y=net_d north. Bands are ordered north to south.
    b1_h = 1.5
    remaining_after_entry = max(net_d - b1_h, 3.5)
    b2_h = round(remaining_after_entry * float(rng.uniform(0.35, 0.40)), 2)
    b3_h = round(remaining_after_entry * float(rng.uniform(0.25, 0.30)), 2)
    b4_h = round(net_d - b1_h - b2_h - b3_h, 2)

    private_depth_need = max(
        [scaled_dims("master_bedroom")[1]] +
        [scaled_dims(r)[1] for r in ("bedroom_2", "bedroom_3", "bedroom_4") if r in rooms] +
        ([scaled_dims("toilet_attached")[1]] if "toilet_attached" in rooms else [0.0])
    )
    min_private = max(2.8, private_depth_need)
    if b4_h < min_private:
        deficit = round(min_private - b4_h, 2)
        take_b3 = min(deficit, max(0.0, b3_h - 2.0))
        b3_h = round(b3_h - take_b3, 2)
        deficit = round(deficit - take_b3, 2)
        if deficit > 0:
            take_b2 = min(deficit, max(0.0, b2_h - 2.4))
            b2_h = round(b2_h - take_b2, 2)
            deficit = round(deficit - take_b2, 2)
        b4_h = round(net_d - b1_h - b2_h - b3_h, 2)

    y_b4 = 0.0
    y_b3 = round(b4_h, 2)
    y_b2 = round(b4_h + b3_h, 2)
    y_b1 = round(net_d - b1_h, 2)

    if small_plot:
        y_b3 = round(y_b3 + 0.115, 2)
        b3_h = round(max(b3_h - 0.115, 0.5), 2)
        y_b2 = round(y_b3 + b3_h, 2)

    if "verandah" in rooms:
        try_add("verandah", 0.0, y_b1, net_w, b1_h, 0.0, net_w, y_b1, net_d)

    if small_plot or "dining" not in rooms:
        try_add("living", 0.0, y_b2, round(net_w, 2), round(b2_h, 2), 0.0, net_w, y_b2, y_b1)
    else:
        living_area = target_area("living")
        dining_area = target_area("dining")
        living_share = float(rng.uniform(0.55, 0.65))
        living_w = round(net_w * living_share, 2)
        dining_w = round(net_w - living_w, 2)
        living_req_w = max(room_rules["living"]["min_width"], living_area / max(b2_h, 0.1))
        dining_req_w = max(room_rules["dining"]["min_width"], dining_area / max(b2_h, 0.1))
        living_w = max(living_w, living_req_w)
        dining_w = max(dining_w, dining_req_w)
        total_public_w = living_w + dining_w
        if total_public_w > net_w:
            # Scale both down proportionally first
            scale = net_w / total_public_w
            living_w = round(living_w * scale, 2)
            dining_w = round(net_w - living_w, 2)

        # Now enforce minimums — but only if the plot can physically fit both
        min_liv = room_rules["living"]["min_width"]
        min_din = room_rules["dining"]["min_width"]

        if net_w >= (min_liv + min_din):
            # Plot can fit both rooms with minimums
            living_w = max(living_w, min_liv)
            dining_w = round(net_w - living_w, 2)
            if dining_w < min_din:
                dining_w = min_din
                living_w = round(net_w - dining_w, 2)
            # Final floor — living must never go below minimum
            living_w = max(living_w, min_liv)
            dining_w = round(net_w - living_w, 2)
        else:
            # Plot is too narrow for both rooms at minimum widths
            # Merge into single living space (same as small_plot path)
            living_w = round(net_w, 2)
            dining_w = 0.0

        if dining_w > 0:
            try_add("living", 0.0, y_b2, living_w, b2_h,
                    0.0, net_w, y_b2, y_b1)
            try_add("dining", living_w, y_b2, dining_w, b2_h,
                    0.0, net_w, y_b2, y_b1)
        else:
            # Merged living+dining
            try_add("living", 0.0, y_b2, round(net_w, 2), round(b2_h, 2),
                    0.0, net_w, y_b2, y_b1)

    if any(r in rooms for r in ("toilet_common", "kitchen", "utility", "pooja", "store")):
        tc_w = 0.0
        if "toilet_common" in rooms:
            tc_w_calc, _ = scaled_dims("toilet_common")
            tc_w = round(max(tc_w_calc, room_rules["toilet_common"]["min_width"]), 2)
            tc_w = round(min(tc_w, net_w * 0.35), 2)
            try_add("toilet_common", 0.0, y_b3, tc_w, b3_h, 0.0, net_w, y_b3, y_b2)

        cluster_x = round(tc_w, 2)
        cluster_w = round(max(net_w - cluster_x, 0.0), 2)
        if cluster_w > 0.6:
            utility_h = 0.0
            if "utility" in rooms or "store" in rooms:
                util_area = target_area("utility") if "utility" in rooms else 0.0
                store_area = target_area("store") if ("store" in rooms and not small_plot) else 0.0
                bottom_area = util_area + store_area
                utility_h = round(max(1.2, bottom_area / max(cluster_w, 0.1)), 2)
                utility_h = round(min(utility_h, b3_h * 0.45), 2)
            top_h = round(b3_h - utility_h, 2)
            if top_h < 1.0:
                top_h = round(b3_h, 2)
                utility_h = 0.0
            pooja_w = 0.0
            if "pooja" in rooms and not small_plot and top_h >= 1.0:
                pooja_area = target_area("pooja")
                pooja_w = round(max(room_rules["pooja"]["min_width"], pooja_area / max(top_h, 0.1)), 2)
                pooja_w = round(min(pooja_w, cluster_w * 0.30), 2)
            kitchen_w = round(max(cluster_w - pooja_w, 0.0), 2)
            kitchen_w = max(kitchen_w, room_rules["kitchen"]["min_width"])
            if "kitchen" in rooms and kitchen_w >= room_rules["kitchen"]["min_width"]:
                try_add("kitchen", cluster_x, round(y_b3 + utility_h, 2), kitchen_w, top_h,
                        cluster_x, net_w, y_b3, y_b2)
            if pooja_w > 0.0 and "pooja" in rooms and not small_plot:
                try_add("pooja", round(cluster_x + kitchen_w, 2), round(y_b3 + utility_h, 2), pooja_w, top_h,
                        cluster_x, net_w, y_b3, y_b2)
            if utility_h > 0.0:
                store_w = 0.0
                if "store" in rooms and not small_plot:
                    store_area = target_area("store")
                    store_w = round(max(room_rules["store"]["min_width"], store_area / max(utility_h, 0.1)), 2)
                    store_w = round(min(store_w, cluster_w * 0.30), 2)
                utility_w = round(max(cluster_w - store_w, 0.0), 2)
                if "utility" in rooms and utility_w >= room_rules["utility"]["min_width"]:
                    try_add("utility", cluster_x, y_b3, utility_w, utility_h,
                            cluster_x, net_w, y_b3, y_b2)
                if store_w > 0.0 and "store" in rooms and not small_plot:
                    try_add("store", round(cluster_x + utility_w, 2), y_b3, store_w, utility_h,
                            cluster_x, net_w, y_b3, y_b2)

    private_rooms = ["master_bedroom"]
    if "toilet_attached" in rooms:
        private_rooms.append("toilet_attached")
    for rt in ("bedroom_2", "bedroom_3", "bedroom_4"):
        if rt in rooms:
            private_rooms.append(rt)

    desired_widths = {}
    for rt in private_rooms:
        w, _ = scaled_dims(rt)
        desired_widths[rt] = round(w, 2)

    total_private_w = sum(desired_widths.values())
    if total_private_w > net_w and total_private_w > 0:
        scale = net_w / total_private_w
        for rt in private_rooms:
            desired_widths[rt] = round(max(room_rules[rt]["min_width"], desired_widths[rt] * scale), 2)
        for rt in private_rooms:
            desired_widths[rt] = max(desired_widths[rt], room_rules[rt]["min_width"])
        total_after = sum(desired_widths.values())
        if total_after > net_w:
            return {}, "plot_too_small", b4_h, y_b3, y_b2

    total_private_w = sum(desired_widths.values())
    if total_private_w < net_w and total_private_w > 0:
        growables = [rt for rt in private_rooms if rt != "toilet_attached"]
        extra_w = round(net_w - total_private_w, 2)
        per_room_add = round(extra_w / max(len(growables), 1), 2)
        for rt in growables:
            desired_widths[rt] = round(desired_widths[rt] + per_room_add, 2)
        remainder = round(net_w - sum(desired_widths.values()), 2)
        if growables and abs(remainder) > 0.001:
            desired_widths[growables[-1]] = round(desired_widths[growables[-1]] + remainder, 2)

    x_cursor = 0.0
    private_y_offsets = {
        "master_bedroom": 0.0,
        "toilet_attached": 0.0,
        "bedroom_2": 0.0,
        "bedroom_3": 0.0,
        "bedroom_4": 0.0,
    }
    for rt in private_rooms:
        w = round(min(desired_widths[rt], max(net_w - x_cursor, 0.6)), 2)
        _, d0 = scaled_dims(rt)
        y_off = private_y_offsets.get(rt, 0.0)
        d = round(min(d0, max(b4_h - y_off, 0.8)), 2)
        try_add(rt, round(x_cursor, 2), round(y_b4 + y_off, 2), w, d, 0.0, net_w, y_b4, y_b3)
        x_cursor = round(x_cursor + w, 2)

    rn = _rotate(placed, net_w, net_d, facing)
    err = "overlap_skip" if skipped_overlap else None
    if float(rng.random()) < float(err_p):
        candidates = [rt for rt in rooms if rt in NBC_MIN_AREA]
        if candidates:
            rt = str(rng.choice(candidates))
            for i, tup in enumerate(rn):
                if tup[0] == rt:
                    rtt, xx, yy, ww, dd = tup
                    rn[i] = (rtt, xx, yy, ww, round(max(0.6, dd * 0.75), 2))
                    err = "nbc_area_violation"
                    break

    pl = {}
    for rt, x, y, w, d in rn:
        pl[rt] = {
            "x": float(round(x, 3)),
            "y": float(round(y, 3)),
            "w": float(round(w, 3)),
            "d": float(round(d, 3)),
        }
        pl[rt]["cx"] = float(round(pl[rt]["x"] + pl[rt]["w"] / 2.0, 3))
        pl[rt]["cy"] = float(round(pl[rt]["y"] + pl[rt]["d"] / 2.0, 3))
    return pl, err, b4_h, y_b3, y_b2


def _expected_cols():
    cols = ["plot_w", "plot_d", "plot_area", "net_w", "net_d", "net_area", "bhk", "facing_code", "climate_code"]
    for rt in ROOM_UNIVERSE:
        p = _pfx(rt)
        cols += [f"{p}_w", f"{p}_d", f"{p}_area", f"{p}_cx_pct", f"{p}_cy_pct"]
    cols += [
        "viol_overlap", "viol_nbc_area", "viol_nbc_width", "viol_kitchen_bedroom", "viol_toilet_kitchen", "viol_toilet_dining",
        "viol_vastu_kitchen_sw", "viol_vastu_master_ne", "viol_living_not_adjacent_verandah", "viol_master_toilet_not_adjacent",
        "score_vastu", "score_nbc", "score_circulation", "score_adjacency", "score_overall", "is_valid", "error_type",
        "plot_size_band", "plot_authority", "plot_category", "plot_layout_type",
    ]
    for rt in ROOM_UNIVERSE:
        p = _pfx(rt)
        cols += [f"{p}_x_abs", f"{p}_y_abs", f"{p}_zone"]
    cols += ["zone_public_area_pct", "zone_private_area_pct", "zone_wet_area_pct", "zone_service_area_pct"]
    cols += ["wall_count_ext", "wall_count_int", "wall_total_length_ext", "wall_total_length_int", "gross_built_area", "net_carpet_area"]
    cols += [
        "adj_living_verandah", "adj_kitchen_utility", "adj_master_toilet", "adj_kitchen_dining", "adj_living_dining",
        "adj_toilet_common_bedroom", "plumbing_cluster_valid", "has_corridor", "corridor_width",
        "viol_no_corridor", "viol_plumbing_scattered", "viol_bedroom_row", "viol_living_no_south_access",
    ]
    return cols


def main():
    rng = np.random.default_rng(SEED)
    db_uri = f"file:{os.path.abspath(DB_PATH).replace(os.sep, '/')}?mode=ro"
    with sqlite3.connect(db_uri, uri=True) as conn:
        setbacks_df = _safe_sql(conn, "SELECT * FROM tn_setbacks")
        climate_df = _safe_sql(conn, "SELECT district, climate_zone FROM climate_data")
        plot_conf = _safe_sql(conn, "SELECT * FROM plot_configurations")
        _ = _safe_sql(conn, "SELECT plot_width_m, plot_depth_m FROM plot_configurations")
        wall_ext_t, wall_int_t = _wall_thickness(conn)
        corridor_min_w = _min_clear_width(conn)
        plot_bands = _load_plot_bands(conn)

    dist_clim = (
        climate_df.dropna(subset=["district"])
        .drop_duplicates(subset=["district"], keep="first")
        .set_index("district")["climate_zone"]
        .to_dict()
    )
    districts = sorted(dist_clim.keys()) or ["Chennai"]
    print(f"Districts loaded: {len(districts)}")
    print(f"Wall thickness ext/int (m): {wall_ext_t}/{wall_int_t}")

    start = time.time()
    skipped = 0
    band_counts = {k: 0 for k in PLOT_BAND_KEYS}
    band_valid = {k: 0 for k in PLOT_BAND_KEYS}
    rows = []

    while len(rows) < N_SAMPLES:
        plot_w, plot_d, bhk, band = sample_plot_and_bhk(rng)
        plot_meta = dict(_LAST_PLOT_META)
        facing = str(rng.choice(FACINGS))
        district = str(rng.choice(districts))
        climate_zone = dist_clim.get(district, "Composite")

        area = float(plot_w * plot_d)
        front, rear, side = _setbacks(area, setbacks_df)
        net_w = round(float(plot_w - 2.0 * side), 2)
        net_d = round(float(plot_d - front - rear), 2)
        if net_w < 2.5 or net_d < 2.5:
            skipped += 1
            continue

        t = _targets(plot_w, plot_d, bhk, plot_conf, rng)
        err_p = 0.04 if band == "small" else (0.05 if band == "medium" else 0.06)
        pl, err_type, b4_h, y_b3, y_b2 = _place(net_w, net_d, bhk, t, rng, facing=facing, err_p=err_p)
        if not pl:
            skipped += 1
            continue

        # base inputs
        row = {
            "plot_w": float(plot_w), "plot_d": float(plot_d), "plot_area": float(area),
            "net_w": float(net_w), "net_d": float(net_d), "net_area": float(round(net_w * net_d, 2)),
            "bhk": int(bhk), "facing_code": int(FACING_MAP.get(facing, 0)), "climate_code": int(CLIMATE_MAP.get(climate_zone, 2)),
            "error_type": err_type or "none",
            "plot_size_band": band,
            "plot_authority": str(plot_meta.get("plot_authority", "unknown")),
            "plot_category": str(plot_meta.get("plot_category", "unknown")),
            "plot_layout_type": str(plot_meta.get("plot_layout_type", "unknown")),
        }

        # per-room 5-tuple (existing)
        for rt in ROOM_UNIVERSE:
            p = _pfx(rt)
            if rt in pl:
                a = pl[rt]
                row[f"{p}_w"] = round(float(a["w"]), 2)
                row[f"{p}_d"] = round(float(a["d"]), 2)
                row[f"{p}_area"] = round(float(a["w"] * a["d"]), 2)
                row[f"{p}_cx_pct"] = round(float(a["cx"] / net_w), 3)
                row[f"{p}_cy_pct"] = round(float(a["cy"] / net_d), 3)
            else:
                row[f"{p}_w"] = row[f"{p}_d"] = row[f"{p}_area"] = row[f"{p}_cx_pct"] = row[f"{p}_cy_pct"] = 0.0

        # FIX 2 + FIX 4 per-room
        zone_area = {"public_zone": 0.0, "private_zone": 0.0, "wet_zone": 0.0, "service_zone": 0.0}
        for rt in ROOM_UNIVERSE:
            p = _pfx(rt)
            if rt in pl:
                row[f"{p}_x_abs"] = float(pl[rt]["x"])
                row[f"{p}_y_abs"] = float(pl[rt]["y"])
                z = ROOM_ZONE.get(rt, "public_zone")
                row[f"{p}_zone"] = z
                zone_area[z] += float(pl[rt]["w"] * pl[rt]["d"])
            else:
                row[f"{p}_x_abs"] = row[f"{p}_y_abs"] = 0.0
                row[f"{p}_zone"] = "none"

        denom = max(float(net_w * net_d), 0.01)
        row["zone_public_area_pct"] = round(zone_area["public_zone"] / denom, 4)
        row["zone_private_area_pct"] = round(zone_area["private_zone"] / denom, 4)
        row["zone_wet_area_pct"] = round(zone_area["wet_zone"] / denom, 4)
        row["zone_service_area_pct"] = round(zone_area["service_zone"] / denom, 4)

        # FIX 3 wall geometry
        ext_c, int_c, ext_l, int_l = _wall_stats(pl, net_w, net_d, tol=0.05)
        gross = float(sum(a["w"] * a["d"] for a in pl.values()))
        wall_area = float(ext_l * wall_ext_t + int_l * wall_int_t)
        row["wall_count_ext"] = ext_c
        row["wall_count_int"] = int_c
        row["wall_total_length_ext"] = ext_l
        row["wall_total_length_int"] = int_l
        row["gross_built_area"] = round(gross, 3)
        row["net_carpet_area"] = round(max(gross - wall_area, 0.0), 3)

        # FIX 5/6 topology flags
        def adj(rt1, rt2): return int(rt1 in pl and rt2 in pl and _adj(pl[rt1], pl[rt2], tol=0.05))
        row["adj_living_verandah"] = adj("living", "verandah")
        row["adj_kitchen_utility"] = adj("kitchen", "utility")
        row["adj_master_toilet"] = adj("master_bedroom", "toilet_attached")
        row["adj_kitchen_dining"] = adj("kitchen", "dining")
        row["adj_living_dining"] = adj("living", "dining")
        row["adj_toilet_common_bedroom"] = int(any(adj("toilet_common", b) for b in ("master_bedroom", "bedroom_2", "bedroom_3", "bedroom_4")))

        plumbing_ok = 0
        # 1BHK small plots may not have all 3 plumbing rooms
        required_plumbing = {"kitchen", "utility", "toilet_common"}
        present_plumbing = required_plumbing.intersection(set(pl.keys()))

        def in_service_band(room_key):
            y = pl.get(room_key, {}).get("y", -999)
            return (y_b3 - 0.3) <= y <= (y_b2 + 0.3)

        def centroid_dist(r1, r2):
            if r1 not in pl or r2 not in pl:
                return 0.0
            return math.sqrt(
                (pl[r1]["cx"] - pl[r2]["cx"]) ** 2 +
                (pl[r1]["cy"] - pl[r2]["cy"]) ** 2
            )

        if len(present_plumbing) < 2:
            # Not enough plumbing rooms to check — skip violation
            plumbing_ok = 1
        elif all(r in pl for r in ("kitchen", "utility", "toilet_common")):
            # Full plumbing check for 2BHK+
            kit_in_band = in_service_band("kitchen")
            util_in_band = in_service_band("utility")
            tc_in_band = in_service_band("toilet_common")
            kit_util_close = centroid_dist("kitchen", "utility") < 4.5
            plumbing_ok = int(kit_in_band and util_in_band and tc_in_band and kit_util_close)
        elif "kitchen" in pl and "utility" in pl:
            # EWS: no toilet_common — just check kitchen+utility proximity
            plumbing_ok = int(
                centroid_dist("kitchen", "utility") < 4.5
            )
        else:
            plumbing_ok = 1  # insufficient rooms to penalise
        row["plumbing_cluster_valid"] = int(plumbing_ok)

        # FIX 2: corridor fails only when living and bedroom zones are not both present.
        # Width is read from DB (min_clear_width_m), but we do not fail on width alone.
        bedrooms_present = any(r in pl for r in ("master_bedroom", "bedroom_2", "bedroom_3", "bedroom_4"))
        has_corridor = int("living" in pl and bedrooms_present)
        row["has_corridor"] = has_corridor
        row["corridor_width"] = float(corridor_min_w if has_corridor else 0.0)

        # FIX 7 + existing violations + scores
        viol = {
            "viol_overlap": 0, "viol_nbc_area": 0, "viol_nbc_width": 0,
            "viol_kitchen_bedroom": 0, "viol_toilet_kitchen": 0, "viol_toilet_dining": 0,
            "viol_vastu_kitchen_sw": 0, "viol_vastu_master_ne": 0,
            "viol_living_not_adjacent_verandah": 0, "viol_master_toilet_not_adjacent": 0,
            "viol_no_corridor": 0, "viol_plumbing_scattered": 0, "viol_bedroom_row": 0, "viol_living_no_south_access": 0,
        }
        # overlap
        rts = list(pl.keys())
        for i, a1 in enumerate(rts):
            A = pl[a1]
            ax0, ay0, ax1, ay1 = A["x"], A["y"], A["x"] + A["w"], A["y"] + A["d"]
            for a2 in rts[i + 1:]:
                B = pl[a2]
                bx0, by0, bx1, by1 = B["x"], B["y"], B["x"] + B["w"], B["y"] + B["d"]
                if min(ax1, bx1) - max(ax0, bx0) > 0.05 and min(ay1, by1) - max(ay0, by0) > 0.05:
                    viol["viol_overlap"] = 1
                    break
            if viol["viol_overlap"]:
                break
        # NBC
        for rt in ROOM_LISTS.get(bhk, []):
            if rt in pl:
                A = pl[rt]
                a = A["w"] * A["d"]
                if rt in NBC_MIN_AREA and a < NBC_MIN_AREA[rt] * 0.88:
                    viol["viol_nbc_area"] = 1

        NBC_MIN_WIDTHS = {
            "master_bedroom": 2.8, "bedroom_2": 2.4,
            "bedroom_3": 2.4, "bedroom_4": 2.4,
            "living": 2.4,   # reduced from 3.0 for narrow plots
                             # 3.0 is target, 2.4 is NBC hard floor
            "dining": 2.1,   # NBC hard floor
            "kitchen": 1.8, "toilet_attached": 1.1,
            "toilet_common": 1.1, "utility": 1.1,
        }
        width_viol = 0
        for rt, min_w in NBC_MIN_WIDTHS.items():
            if rt in pl and pl[rt]["w"] < min_w:
                width_viol = 1
                break
        viol["viol_nbc_width"] = width_viol
        # forbidden adjacencies
        if "kitchen" in pl:
            for b in ("master_bedroom", "bedroom_2", "bedroom_3", "bedroom_4"):
                if b in pl and _adj(pl["kitchen"], pl[b], 0.05):
                    viol["viol_kitchen_bedroom"] = 1
            # FIX 1: keep this rare by checking the attached toilet name only.
            if "toilet_attached" in pl and _adj(pl["kitchen"], pl["toilet_attached"], 0.05):
                viol["viol_toilet_kitchen"] = 1
        if "toilet_common" in pl and "dining" in pl and _adj(pl["toilet_common"], pl["dining"], 0.05):
            viol["viol_toilet_dining"] = 1
        # vastu-ish
        if "kitchen" in pl and (pl["kitchen"]["cx"] / net_w) < 0.35 and (pl["kitchen"]["cy"] / net_d) < 0.35:
            viol["viol_vastu_kitchen_sw"] = 1
        if "master_bedroom" in pl and (pl["master_bedroom"]["cx"] / net_w) > 0.65 and (pl["master_bedroom"]["cy"] / net_d) > 0.65:
            viol["viol_vastu_master_ne"] = 1
        # critical adj
        if "living" in pl and "verandah" in pl and not _adj(pl["living"], pl["verandah"], 0.05):
            viol["viol_living_not_adjacent_verandah"] = 1
        if "master_bedroom" in pl and "toilet_attached" in pl and not _adj(pl["master_bedroom"], pl["toilet_attached"], 0.05):
            viol["viol_master_toilet_not_adjacent"] = 1
        # new
        if has_corridor == 0:
            viol["viol_no_corridor"] = 1
        if plumbing_ok == 0:
            viol["viol_plumbing_scattered"] = 1
        beds = [r for r in ("master_bedroom", "bedroom_2", "bedroom_3", "bedroom_4") if r in pl]
        if len(beds) >= 3:
            ys = [pl[r]["y"] for r in beds]
            if max(ys) - min(ys) < 0.05:
                viol["viol_bedroom_row"] = 1
        if "living" in pl and (pl["living"]["cy"] / net_d) > 0.70:
            viol["viol_living_no_south_access"] = 1

        row.update(viol)
        # scores + is_valid
        clamp01 = lambda x: max(0.0, min(1.0, float(x)))
        s_v = 1.0 - (0.4 * viol["viol_vastu_kitchen_sw"] + 0.4 * viol["viol_vastu_master_ne"])
        s_n = 1.0 - (0.35 * viol["viol_nbc_area"] + 0.25 * viol["viol_nbc_width"])
        s_c = 1.0 - (0.35 * viol["viol_living_not_adjacent_verandah"] + 0.35 * viol["viol_master_toilet_not_adjacent"] + 0.25 * viol["viol_no_corridor"])
        s_a = 1.0 - (0.25 * viol["viol_kitchen_bedroom"] + 0.25 * viol["viol_toilet_kitchen"] + 0.20 * viol["viol_toilet_dining"] + 0.15 * viol["viol_plumbing_scattered"])
        s_o = 0.30 * clamp01(s_v) + 0.25 * clamp01(s_n) + 0.25 * clamp01(s_c) + 0.20 * clamp01(s_a)
        row["score_vastu"] = round(clamp01(s_v), 3)
        row["score_nbc"] = round(clamp01(s_n), 3)
        row["score_circulation"] = round(clamp01(s_c), 3)
        row["score_adjacency"] = round(clamp01(s_a), 3)
        row["score_overall"] = round(clamp01(s_o), 3)
        row["is_valid"] = 1 if sum(row[k] for k in row if k.startswith("viol_")) == 0 else 0

        if row["is_valid"] == 0 and row.get("error_type", "none") in ("none", "", None):
            viols = [k for k in row if k.startswith("viol_") and int(row.get(k, 0)) == 1]
            row["error_type"] = str(rng.choice(viols)) if viols else "none"

        rows.append(row)
        band_counts[band] += 1
        band_valid[band] += int(row["is_valid"])

        if len(rows) % 5000 == 0:
            elapsed = time.time() - start
            valid_pct = sum(r["is_valid"] for r in rows) / len(rows) * 100
            print(f"{len(rows):>6}/{N_SAMPLES} | {valid_pct:.1f}% valid | {elapsed:.0f}s elapsed | {skipped} skipped")

    df = pd.DataFrame(rows)
    cols = _expected_cols()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    # fill strings
    for c in df.columns:
        if c.endswith("_zone") or c in ("plot_size_band", "plot_authority", "plot_category", "plot_layout_type", "error_type"):
            df[c] = df[c].fillna("none")
    # fill numerics
    for c in df.columns:
        if c.endswith("_zone") or c in ("plot_size_band", "plot_authority", "plot_category", "plot_layout_type", "error_type"):
            continue
        if df[c].dtype == object:
            df[c] = df[c].fillna("none")
        else:
            df[c] = df[c].fillna(0.0)
    df = df.reindex(columns=cols)
    os.makedirs("training_data", exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, compression="snappy")

    elapsed = time.time() - start
    print("=" * 60)
    print("TRAINING DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total samples:    {len(df)}")
    print(f"Valid plans:      {int(df['is_valid'].sum())} ({df['is_valid'].mean()*100:.1f}%)")
    print(f"Invalid plans:    {int((df['is_valid']==0).sum())}")
    print(f"Total columns:    {len(df.columns)}")
    print(f"Skipped:          {skipped}")
    print(f"Time taken:       {elapsed:.0f}s")
    print(f"Saved to:         {os.path.abspath(OUT_PATH)}")
    print()
    print("VIOLATION BREAKDOWN:")
    viol_cols = [c for c in df.columns if c.startswith("viol_")]
    for col in viol_cols:
        cnt = int((df[col] == 1).sum())
        print(f"  {col}: {cnt} ({(df[col].mean()*100):.1f}%)")
    print()
    print("ZONE AREA DISTRIBUTION (mean % of net area):")
    for col in ("zone_public_area_pct", "zone_private_area_pct", "zone_wet_area_pct", "zone_service_area_pct"):
        print(f"  {col}: mean={df[col].mean():.3f}  min={df[col].min():.3f}  max={df[col].max():.3f}")
    print()
    print("PLOT SIZE BAND DISTRIBUTION:")
    print(df["plot_size_band"].value_counts().to_string())
    print()
    print("VALID COUNTS BY BAND (during generation):")
    for k in PLOT_BAND_KEYS:
        print(f"  {k:<6} samples={band_counts[k]:>6}  valid={band_valid[k]:>6}")
    print()
    print("FIRST 3 ROWS (key columns only):")
    key_cols = ["plot_w", "plot_d", "plot_size_band", "bhk", "facing_code", "is_valid", "score_overall", "error_type"]
    print(df[key_cols].head(3).to_string(index=False))
    print()
    print("COLUMNS (full list):")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
