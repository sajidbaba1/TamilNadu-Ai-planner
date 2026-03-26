"""
tests/test_backend.py
=====================
Backend validation for generated floor plans.
Tests generate_plan() for 8 canonical Tamil Nadu plot combinations without
touching the renderer or the Streamlit app.

Run:
    python tests/test_backend.py
"""

import sys, os
# Force UTF-8 output on Windows (default charmap can't encode ✓ ✗ ↔)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.engine_api import generate_plan
from engine.engine import NBC_MIN_AREA, NBC_MIN_WIDTH

# ─── geometry tolerance ────────────────────────────────────────────────────────
# Interior wall gap = 0.115 m; WALL_TOL covers that plus a small margin.
WALL_TOL = 0.14

# ─── adjacency rules ──────────────────────────────────────────────────────────
# (room_a, room_b, display_label)
REQUIRED_WALL = [
    ("living",         "verandah",        "living ↔ verandah"),
    ("kitchen",        "utility",         "kitchen ↔ utility"),
    ("master_bedroom", "toilet_attached", "master ↔ toilet_att"),
    ("kitchen",        "dining",          "kitchen ↔ dining"),
]
FORBIDDEN_WALL = [
    ("toilet_common", "dining",   "toilet_common ↔ dining"),
    ("toilet_common", "kitchen",  "toilet_common ↔ kitchen"),
]

# ─── vastu ────────────────────────────────────────────────────────────────────
# Mirrors engine score_and_explain() logic exactly so our report agrees with
# fp.score_vastu.
#   Kitchen in SW corner (cx<0.35 AND cy<0.35)  → bad
#   Master  in NE corner (cx>0.65 AND cy>0.65)  → bad
#   Pooja   NOT in NE quad (cx>0.55 AND cy>0.55) → bad
#   Toilet  compass == NE                         → bad
VASTU_KITCHEN_GOOD  = {"NE", "SE", "E", "N", "C", "S", "W", "NW"}  # all except explicit SW trap
VASTU_MASTER_GOOD   = {"SW", "S", "W", "SE", "C", "NW"}            # all except explicit NE trap
VASTU_TOILET_BAD    = {"NE"}

# ─── geometry helpers ─────────────────────────────────────────────────────────

def _rect(r):
    """(x0, y0, x1, y1) bounding box of Room r."""
    return r.x, r.y, r.x + r.width, r.y + r.depth


def share_wall(a, b, tol=WALL_TOL):
    """True if rooms a and b share a wall (separated by ≤ tol)."""
    ax0, ay0, ax1, ay1 = _rect(a)
    bx0, by0, bx1, by1 = _rect(b)
    # vertical gap
    if abs(ax1 - bx0) <= tol or abs(bx1 - ax0) <= tol:
        if min(ay1, by1) - max(ay0, by0) > tol:
            return True
    # horizontal gap
    if abs(ay1 - by0) <= tol or abs(by1 - ay0) <= tol:
        if min(ax1, bx1) - max(ax0, bx0) > tol:
            return True
    return False


def rooms_overlap(a, b, tol=0.05):
    """True if rooms a and b physically overlap (not just share a wall)."""
    ax0, ay0, ax1, ay1 = _rect(a)
    bx0, by0, bx1, by1 = _rect(b)
    return (ax0 < bx1 - tol and ax1 > bx0 + tol and
            ay0 < by1 - tol and ay1 > by0 + tol)

# ─── display helpers ──────────────────────────────────────────────────────────

def ck(ok):
    return "✓" if ok else "✗"


def _na(reason=""):
    return f"N/A{(' (' + reason + ')') if reason else ''}"


# ─── per-floor report ─────────────────────────────────────────────────────────

def report_floor(label, fp, scores, score_prefix=""):
    """
    Print a full architectural validation report for one FloorPlan floor.
    Returns a dict of failure counts for the summary table.
    """
    rooms   = fp.rooms          # dict[room_type, Room]  (API contract)
    net_w   = fp.net_w
    net_d   = fp.net_d
    facing  = fp.facing
    rlist   = list(rooms.values())

    sv  = scores.get(f"{score_prefix}score_valid",       0.0)
    so  = scores.get(f"{score_prefix}score_overall",     0.0)
    sva = scores.get(f"{score_prefix}score_vastu",       0.0)
    sn  = scores.get(f"{score_prefix}score_nbc",         0.0)
    sc  = scores.get(f"{score_prefix}score_circulation", 0.0)
    sa  = scores.get(f"{score_prefix}score_adjacency",   0.0)

    nbc_fails  = 0
    adj_fails  = 0
    zone_fails = 0
    overlap_pairs = []

    print(f"\n{'=' * 60}")
    print(f"=== {label} ===")
    print(f"{'=' * 60}")
    print(f"Plot: {fp.plot_w}x{fp.plot_d}m  |  "
          f"Net: {net_w}x{net_d}m  |  "
          f"BHK: {fp.bhk}  |  Facing: {facing}")
    print(f"Score: overall={so:.3f}  valid={sv:.3f}  "
          f"vastu={sva:.3f}  nbc={sn:.3f}  "
          f"circ={sc:.3f}  adj={sa:.3f}")

    # ── ROOM PLACEMENT ────────────────────────────────────────────────────────
    print("\nROOM PLACEMENT:")
    print(f"  {'room':<22} {'x':>6} {'y':>6}  {'w':>6} {'d':>6}  {'area':>6}  "
          f"{'NBC_min':>7}  STATUS")
    for rt, r in rooms.items():
        nbc_min = NBC_MIN_AREA.get(rt)
        nbc_str = f"{nbc_min:.1f}" if nbc_min else "--"
        ok = (r.area >= nbc_min * 0.88) if nbc_min else True
        if not ok:
            nbc_fails += 1
        status = "OK  " if ok else "FAIL"
        print(f"  {rt:<22} {r.x:>6.2f} {r.y:>6.2f}  "
              f"{r.width:>6.2f} {r.depth:>6.2f}  {r.area:>6.1f}  "
              f"{nbc_str:>7}  {status} {ck(ok)}")

    # ── ADJACENCY CHECK ───────────────────────────────────────────────────────
    print("\nADJACENCY CHECK:")
    for ra, rb, lbl in REQUIRED_WALL:
        if ra in rooms and rb in rooms:
            ok = share_wall(rooms[ra], rooms[rb])
            if not ok:
                adj_fails += 1
            tag = "SHARE_WALL" if ok else "MISSING   "
            print(f"  {lbl:<32} {tag}  {ck(ok)}")
        else:
            absent = ra if ra not in rooms else rb
            print(f"  {lbl:<32} {_na(absent + ' not in program')}")

    for ra, rb, lbl in FORBIDDEN_WALL:
        if ra in rooms and rb in rooms:
            touching = share_wall(rooms[ra], rooms[rb])
            ok = not touching
            if not ok:
                adj_fails += 1
            tag = "SEPARATED " if ok else "TOUCHING! "
            print(f"  {lbl:<32} {tag}  {ck(ok)}  (must not touch)")
        else:
            absent = ra if ra not in rooms else rb
            print(f"  {lbl:<32} {_na(absent + ' not in program')}")

    # ── ZONE CHECK ────────────────────────────────────────────────────────────
    print("\nZONE CHECK:")

    # Verandah on road-facing wall
    if "verandah" in rooms:
        r = rooms["verandah"]
        if facing == "N":
            on_road = (r.y + r.depth) >= (net_d - 0.30)
        elif facing == "S":
            on_road = r.y <= 0.30
        elif facing == "E":
            on_road = (r.x + r.width) >= (net_w - 0.30)
        else:  # W
            on_road = r.x <= 0.30
        if not on_road:
            zone_fails += 1
        print(f"  Verandah on {facing}-facing wall:  "
              f"{'YES' if on_road else 'NO ':3}  {ck(on_road)}  "
              f"(x={r.x:.2f} y={r.y:.2f} w={r.width:.2f} d={r.depth:.2f})")
    else:
        print(f"  Verandah on road-facing wall:  {_na()}")

    # Kitchen compass / SW trap
    if "kitchen" in rooms:
        r = rooms["kitchen"]
        sw_trap = r.cx_pct < 0.35 and r.cy_pct < 0.35
        ok = not sw_trap
        if not ok:
            zone_fails += 1
        print(f"  Kitchen not in SW trap:        "
              f"{'YES' if ok else 'NO ':3}  {ck(ok)}  "
              f"(compass={r.compass}  cx={r.cx_pct:.2f} cy={r.cy_pct:.2f})")
    else:
        print(f"  Kitchen compass:               {_na()}")

    # Master bedroom compass / NE trap
    if "master_bedroom" in rooms:
        r = rooms["master_bedroom"]
        ne_trap = r.cx_pct > 0.65 and r.cy_pct > 0.65
        ok = not ne_trap
        if not ok:
            zone_fails += 1
        print(f"  Master not in NE trap:         "
              f"{'YES' if ok else 'NO ':3}  {ck(ok)}  "
              f"(compass={r.compass}  cx={r.cx_pct:.2f} cy={r.cy_pct:.2f})")
    else:
        print(f"  Master bedroom compass:        {_na()}")

    # Pooja in NE quadrant
    if "pooja" in rooms:
        r = rooms["pooja"]
        ok = r.cx_pct > 0.55 and r.cy_pct > 0.55
        if not ok:
            zone_fails += 1
        print(f"  Pooja in NE quadrant:          "
              f"{'YES' if ok else 'NO ':3}  {ck(ok)}  "
              f"(cx={r.cx_pct:.2f} cy={r.cy_pct:.2f})")
    else:
        print(f"  Pooja in NE quadrant:          {_na('not in this BHK program')}")

    # Toilets not in NE (sacred Ishanya zone)
    for toilet in ("toilet_attached", "toilet_common"):
        if toilet in rooms:
            r = rooms[toilet]
            ok = r.compass not in VASTU_TOILET_BAD
            if not ok:
                zone_fails += 1
            short = toilet.replace("toilet_", "wc_")
            print(f"  {short} not in NE:           "
                  f"{'YES' if ok else 'NO ':3}  {ck(ok)}  "
                  f"(compass={r.compass})")

    # ── NBC WIDTH CHECK ───────────────────────────────────────────────────────
    print("\nNBC WIDTH CHECK:")
    for rt, min_w in NBC_MIN_WIDTH.items():
        if rt in rooms:
            r = rooms[rt]
            ok = r.width >= min_w - 0.01   # 1 cm rounding tolerance
            if not ok:
                nbc_fails += 1
            print(f"  {rt:<22}  w={r.width:.2f} >= {min_w:.2f}  {ck(ok)}")

    # ── CIRCULATION CHECK ─────────────────────────────────────────────────────
    print("\nCIRCULATION CHECK:")
    bedrooms_present = any(rt in rooms for rt in
                           ("master_bedroom", "bedroom_2", "bedroom_3", "bedroom_4"))
    has_corridor = "living" in rooms and bedrooms_present
    print(f"  has_corridor:        {'YES' if has_corridor else 'NO':3}  {ck(has_corridor)}")
    print(f"  corridor_width:      "
          f"{'0.60m >= 0.60m' if has_corridor else 'N/A   '}"
          f"  {ck(has_corridor)}")

    is_first = score_prefix == "first_"
    stair_rt = "staircase_head" if is_first else "staircase"
    if stair_rt in rooms:
        r = rooms[stair_rt]
        print(f"  {stair_rt:<20} PRESENT  {ck(True)}  "
              f"(x={r.x:.2f} y={r.y:.2f} {r.width:.2f}x{r.depth:.2f}m)")
    else:
        if is_first:
            print(f"  staircase_head:      MISSING  {ck(False)}")
        else:
            print(f"  staircase:           N/A (G-only plan)")

    # ── OVERLAP CHECK ─────────────────────────────────────────────────────────
    print("\nOVERLAP CHECK:")
    for i, a in enumerate(rlist):
        for b in rlist[i + 1:]:
            if rooms_overlap(a, b):
                overlap_pairs.append((a.room_type, b.room_type))

    if overlap_pairs:
        print(f"  OVERLAPPING PAIRS FOUND  {ck(False)}")
        for ra, rb in overlap_pairs:
            print(f"    {ra}  ↔  {rb}")
    else:
        print(f"  All {len(rlist)} rooms: NO OVERLAPS  {ck(True)}")

    # ── VASTU SUMMARY ─────────────────────────────────────────────────────────
    print("\nVASTU CHECK:")
    if "kitchen" in rooms:
        r = rooms["kitchen"]
        sw = r.cx_pct < 0.35 and r.cy_pct < 0.35
        print(f"  Kitchen direction:       {r.compass:<4}  {ck(not sw)}  "
              f"(good: NE SE E — worst: SW)")
    if "master_bedroom" in rooms:
        r = rooms["master_bedroom"]
        ne = r.cx_pct > 0.65 and r.cy_pct > 0.65
        print(f"  Master bedroom:          {r.compass:<4}  {ck(not ne)}  "
              f"(good: SW S W — worst: NE)")
    for toilet in ("toilet_attached", "toilet_common"):
        if toilet in rooms:
            r = rooms[toilet]
            ok = r.compass != "NE"
            print(f"  {toilet:<22} {r.compass:<4}  {ck(ok)}  "
                  f"(must not be NE)")
    if "pooja" in rooms:
        r = rooms["pooja"]
        ok = r.cx_pct > 0.55 and r.cy_pct > 0.55
        print(f"  Pooja NE quadrant:       {r.compass:<4}  {ck(ok)}")
    else:
        print(f"  Pooja:                   N/A")

    print("-" * 60)

    return {
        "rooms":    len(rooms),
        "overlaps": len(overlap_pairs),
        "nbc_fail": nbc_fails,
        "adj_fail": adj_fails,
        "zone_fail": zone_fails,
    }


# ─── main ─────────────────────────────────────────────────────────────────────

TEST_CASES = [
    (6,  9,  1, "N", "Trichy",     1),   # EWS minimum
    (9,  12, 2, "N", "Salem",      1),   # Standard small
    (12, 15, 2, "N", "Coimbatore", 1),   # Most common TN plot
    (12, 15, 2, "N", "Coimbatore", 2),   # Same + G+1
    (15, 15, 3, "N", "Chennai",    1),   # Premium square
    (15, 20, 3, "S", "Chennai",    1),   # South facing
    (15, 20, 4, "E", "Madurai",    1),   # East facing 4BHK
    (20, 25, 4, "N", "Coimbatore", 2),   # Large G+1
]


def main():
    print("BACKEND VALIDATION — 8 CANONICAL TEST CASES")
    print("=" * 60)

    summary_rows = []   # (case_label, metrics_dict)
    all_pass = True

    for pw, pd, bhk, facing, district, floors in TEST_CASES:
        floor_label = f"G+{floors - 1}"
        case_key    = f"{pw}x{pd} {bhk}BHK {facing} {district} {floor_label}"

        try:
            result  = generate_plan(pw, pd, bhk, facing, district, floors=floors)
            g       = result["ground"]
            f       = result["first"]   # None if floors == 1
            scores  = result["scores"]

            # Ground floor report
            g_label = f"{pw}x{pd} {bhk}BHK {facing} {district} G"
            g_stats = report_floor(g_label, g, scores, score_prefix="")
            summary_rows.append((g_label, g_stats))

            # First floor report (G+1 only)
            if f is not None:
                f_label = f"{pw}x{pd} {bhk}BHK {facing} {district} G+1 (first floor)"
                f_stats = report_floor(f_label, f, scores, score_prefix="first_")
                summary_rows.append((f_label, f_stats))

        except Exception as exc:
            import traceback
            print(f"\n[FAIL] {case_key}")
            traceback.print_exc()
            summary_rows.append((case_key, {"error": str(exc)}))
            all_pass = False

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("BACKEND VALIDATION SUMMARY")
    print(f"{'=' * 80}")
    hdr = f"{'Case':<42} {'Rooms':>5}  {'Overlaps':>8}  {'NBC_fail':>8}  {'Adj_fail':>8}  {'Zone_fail':>9}  Overall"
    print(hdr)
    print("-" * 80)

    for case_label, stats in summary_rows:
        if "error" in stats:
            print(f"  {case_label:<42}  ERROR: {stats['error'][:30]}")
            all_pass = False
            continue

        rooms     = stats["rooms"]
        overlaps  = stats["overlaps"]
        nbc_fail  = stats["nbc_fail"]
        adj_fail  = stats["adj_fail"]
        zone_fail = stats["zone_fail"]
        any_fail  = overlaps or nbc_fail or adj_fail or zone_fail
        verdict   = "FAIL" if any_fail else "PASS"
        if any_fail:
            all_pass = False

        print(f"  {case_label:<42} {rooms:>5}  {overlaps:>8}  "
              f"{nbc_fail:>8}  {adj_fail:>8}  {zone_fail:>9}  {verdict}")

        if any_fail:
            reasons = []
            if overlaps:   reasons.append(f"{overlaps} overlap pair(s)")
            if nbc_fail:   reasons.append(f"{nbc_fail} NBC violation(s)")
            if adj_fail:   reasons.append(f"{adj_fail} adjacency violation(s)")
            if zone_fail:  reasons.append(f"{zone_fail} zone/vastu violation(s)")
            print(f"    └─ FAILURES: {', '.join(reasons)}")

    print("-" * 80)
    print(f"\n{'ALL PASS' if all_pass else 'FAILURES PRESENT — see detail above'}")


if __name__ == "__main__":
    main()
