"""
Renderer Data Contract Audit
=============================
Calls generate_plan() for all 8 canonical cases and inspects every field
that a professional renderer needs. Does not render anything. Prints a
complete per-floor data audit (what exists, what is missing, what is
geometrically invalid) followed by a cross-case summary table.

Run:  python tests/test_renderer_contract.py

Field-name mapping (dataclass field → renderer-facing name used in audit):
  DoorOpening.position    → x_on_wall
  DoorOpening.swing_into  → swing_direction
  WindowOpening.position  → x_on_wall
  WindowOpening.sill_height → sill_depth
  fp.setback_front/rear/side → fp.setbacks (printed as dict)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.engine_api import generate_plan

CASES = [
    (6,  9,  1, "N", "Trichy",     1),
    (9,  12, 2, "N", "Salem",      1),
    (12, 15, 2, "N", "Coimbatore", 1),
    (15, 15, 3, "N", "Chennai",    1),
    (15, 20, 3, "S", "Chennai",    1),
    (15, 20, 4, "E", "Madurai",    1),
    (20, 25, 4, "N", "Coimbatore", 1),
    (20, 25, 4, "N", "Coimbatore", 2),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v, decimals=3):
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _wall_facing(wall):
    """H walls are top/bottom edges (E/W face); V walls are left/right (N/S face)."""
    if wall is None:
        return "?"
    return "H(E/W)" if getattr(wall, "direction", "?") == "H" else "V(N/S)"


def _rooms_overlap(a, b, tol=1e-3):
    """True if bounding boxes overlap beyond a shared edge."""
    return (a.x + tol < b.x + b.width  and a.x + a.width - tol > b.x and
            a.y + tol < b.y + b.depth   and a.y + a.depth - tol > b.y)


def _rooms_list(fp):
    return list(fp.rooms.values()) if isinstance(fp.rooms, dict) else list(fp.rooms)


def _net_bounds(fp):
    """For E/W-facing plans, _rotate() maps internal y->rendered x, so the
    rendered coordinate space is [0..net_d] x [0..net_w] (axes swapped).
    For N/S-facing it stays [0..net_w] x [0..net_d]."""
    if getattr(fp, "facing", "N") in ("E", "W"):
        return fp.net_d, fp.net_w   # x_bound, y_bound
    return fp.net_w, fp.net_d


# ---------------------------------------------------------------------------
# Per-floor audit — returns a dict of counts for the summary table
# ---------------------------------------------------------------------------

def audit_floor(fp, label):
    """
    Print the full 7-section audit for one FloorPlan and return a stats dict:
      rooms, walls, doors, windows,
      invalid_geom, zero_len_walls, oob_walls,
      missing_door_pos, missing_swing,
      missing_win_pos, missing_win_facing,
      oob_rooms, overlap_rooms, missing_fields (list of strings)
    """
    net_w = fp.net_w
    net_d = fp.net_d
    x_bound, y_bound = _net_bounds(fp)   # accounts for E/W axis swap
    rooms_list = _rooms_list(fp)

    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")

    # -------------------------------------------------------------------
    # SECTION 1: ROOM BOUNDING BOXES
    # -------------------------------------------------------------------
    print("\n--- SECTION 1: ROOM BOUNDING BOXES ---")
    hdr = f"  {'room_type':<22} {'x':>6} {'y':>6} {'w':>6} {'d':>6} {'area':>7}  valid_geometry"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    invalid_geom = []
    for r in rooms_list:
        ok = r.width > 0 and r.depth > 0 and r.x >= 0 and r.y >= 0
        if not ok:
            invalid_geom.append(r.room_type)
        flag = "True" if ok else "FALSE <- invalid"
        print(f"  {r.room_type:<22} {r.x:>6.2f} {r.y:>6.2f} {r.width:>6.2f} "
              f"{r.depth:>6.2f} {r.area:>7.2f}  {flag}")

    print(f"\n  Total rooms: {len(rooms_list)}", end="")
    if invalid_geom:
        print(f"  |  Invalid geometry: {invalid_geom}")
    else:
        print("  |  All valid.")

    # -------------------------------------------------------------------
    # SECTION 2: WALL NETWORK
    # -------------------------------------------------------------------
    print("\n--- SECTION 2: WALL NETWORK ---")
    has_walls = hasattr(fp, "walls") and bool(fp.walls)
    print(f"  Does fp have a 'walls' attribute?          {'YES' if has_walls else 'NO'}")

    zero_len_walls = 0
    oob_walls = 0

    if has_walls:
        tol = 0.01
        print(f"  Wall count:                              {len(fp.walls)}")
        wh = (f"\n  {'wall_type':<10} {'x1':>7} {'y1':>7} {'x2':>7} {'y2':>7} "
              f"{'thick':>6} {'length':>8}")
        print(wh)
        print("  " + "-" * (len(wh) - 3))
        for w in fp.walls:
            length = round(w.length, 4)
            if length == 0:
                zero_len_walls += 1
            out = (w.x1 < -tol or w.x1 > x_bound + tol or
                   w.x2 < -tol or w.x2 > x_bound + tol or
                   w.y1 < -tol or w.y1 > y_bound + tol or
                   w.y2 < -tol or w.y2 > y_bound + tol)
            if out:
                oob_walls += 1
            flag = " <- ZERO" if length == 0 else (" <- OOB" if out else "")
            print(f"  {w.wall_type:<10} {w.x1:>7.3f} {w.y1:>7.3f} "
                  f"{w.x2:>7.3f} {w.y2:>7.3f} {w.thickness:>6.3f} {length:>8.4f}{flag}")
        print(f"\n  Zero-length walls (crash risk):          "
              f"{'YES  count=' + str(zero_len_walls) if zero_len_walls else 'NO'}")
        print(f"  Walls outside net boundary:              "
              f"{'YES  count=' + str(oob_walls) if oob_walls else 'NO'}")

    # -------------------------------------------------------------------
    # SECTION 3: DOOR DATA
    # -------------------------------------------------------------------
    print("\n--- SECTION 3: DOOR DATA ---")
    has_doors = hasattr(fp, "doors") and fp.doors is not None
    print(f"  Does fp have a 'doors' attribute?          {'YES' if has_doors else 'NO'}")

    missing_door_pos = 0
    missing_swing = 0

    if has_doors:
        print(f"  Door count:                              {len(fp.doors)}")
        dh = (f"\n  {'room_from':<22} {'room_to':<22} "
              f"{'wall_side':<10} {'x_on_wall':>10} {'width':>6} swing_direction")
        print(dh)
        print("  " + "-" * (len(dh) - 3))
        for d in fp.doors:
            wall_side = _wall_facing(d.wall) if d.wall else "None"
            pos       = getattr(d, "position", None)
            swing     = getattr(d, "swing_into", None)
            if pos is None or pos == 0:
                missing_door_pos += 1
            if not swing:
                missing_swing += 1
            pos_s   = _fmt(pos) if pos is not None else "None <-"
            swing_s = str(swing) if swing else "None <-"
            print(f"  {d.room_from:<22} {d.room_to:<22} "
                  f"{wall_side:<10} {pos_s:>10} {d.width:>6.2f} {swing_s}")
        print(f"\n  Missing x_on_wall (None or 0):           {missing_door_pos}")
        print(f"  Missing swing_direction:                  {missing_swing}")

    # -------------------------------------------------------------------
    # SECTION 4: WINDOW DATA
    # -------------------------------------------------------------------
    print("\n--- SECTION 4: WINDOW DATA ---")
    has_windows = hasattr(fp, "windows") and fp.windows is not None
    print(f"  Does fp have a 'windows' attribute?        {'YES' if has_windows else 'NO'}")

    missing_win_pos    = 0
    missing_win_facing = 0

    if has_windows:
        print(f"  Window count:                            {len(fp.windows)}")
        wnh = (f"\n  {'room':<22} {'wall_facing':<12} {'x_on_wall':>10} "
               f"{'width':>6} {'sill_depth':>10} is_vent")
        print(wnh)
        print("  " + "-" * (len(wnh) - 3))
        for win in fp.windows:
            wf  = _wall_facing(win.wall) if win.wall else "None"
            pos = getattr(win, "position", None)
            sl  = getattr(win, "sill_height", None)
            if pos is None:
                missing_win_pos += 1
            if win.wall is None:
                missing_win_facing += 1
            wf_s  = wf  if win.wall else "None <-"
            pos_s = _fmt(pos) if pos is not None else "None <-"
            sl_s  = _fmt(sl)  if sl  is not None else "None"
            print(f"  {win.room_type:<22} {wf_s:<12} {pos_s:>10} "
                  f"{win.width:>6.2f} {sl_s:>10} {win.is_ventilator}")
        print(f"\n  Missing x_on_wall:                       {missing_win_pos}")
        print(f"  Missing wall_facing:                     {missing_win_facing}")

    # -------------------------------------------------------------------
    # SECTION 5: METADATA
    # -------------------------------------------------------------------
    print("\n--- SECTION 5: METADATA ---")
    district = getattr(fp, "district",    None)
    climate  = getattr(fp, "climate_zone", None)
    sf       = getattr(fp, "setback_front", None)
    sr       = getattr(fp, "setback_rear",  None)
    ss       = getattr(fp, "setback_side",  None)
    materials = getattr(fp, "materials",       None)
    baker     = getattr(fp, "baker_principles", None)

    print(f"  fp.plot_w:           {fp.plot_w}")
    print(f"  fp.plot_d:           {fp.plot_d}")
    print(f"  fp.net_w:            {fp.net_w}")
    print(f"  fp.net_d:            {fp.net_d}")
    print(f"  fp.facing:           {fp.facing or 'MISSING'}")
    print(f"  fp.district:         {district  or 'MISSING'}")
    print(f"  fp.climate_zone:     {climate   or 'MISSING'}")

    if any(v is not None for v in (sf, sr, ss)):
        print(f"  fp.setbacks:         {{front: {sf}, rear: {sr}, side: {ss}}}")
    else:
        print("  fp.setbacks:         MISSING")

    if materials:
        name = materials[0].get("material_name", materials[0]) if isinstance(materials[0], dict) else materials[0]
        print(f"  fp.materials:        exists — first: {name}")
    else:
        print("  fp.materials:        MISSING or empty")

    if baker:
        print(f"  fp.baker_principles: exists — count: {len(baker)}")
    else:
        print("  fp.baker_principles: MISSING or empty")

    # -------------------------------------------------------------------
    # SECTION 6: MISSING DATA SUMMARY
    # -------------------------------------------------------------------
    print("\n--- SECTION 6: MISSING DATA SUMMARY ---")
    missing_fields = []

    if not has_walls or not fp.walls:
        missing_fields.append("fp.walls is missing or empty")
    if zero_len_walls:
        missing_fields.append(f"{zero_len_walls} wall(s) have zero length (renderer crash risk)")
    if oob_walls:
        missing_fields.append(f"{oob_walls} wall(s) are outside the net boundary")
    if has_doors and missing_door_pos:
        missing_fields.append(f"{missing_door_pos} door(s) have no x_on_wall (position=None or 0)")
    if has_doors and missing_swing:
        missing_fields.append(f"{missing_swing} door(s) have no swing_direction")
    if has_windows and missing_win_pos:
        missing_fields.append(f"{missing_win_pos} window(s) have no x_on_wall (position=None)")
    if has_windows and missing_win_facing:
        missing_fields.append(f"{missing_win_facing} window(s) have no wall_facing (wall=None)")
    if sf is None and sr is None and ss is None:
        missing_fields.append("fp.setbacks (setback_front/rear/side) all missing")
    if not fp.facing:
        missing_fields.append("fp.facing is missing")
    if not district:
        missing_fields.append("fp.district is missing")
    if not climate:
        missing_fields.append("fp.climate_zone is missing")
    if not materials:
        missing_fields.append("fp.materials is missing or empty")
    if not baker:
        missing_fields.append("fp.baker_principles is missing or empty")

    if missing_fields:
        for item in missing_fields:
            print(f"  MISSING: {item}")
    else:
        print("  All required renderer fields are present.")

    # -------------------------------------------------------------------
    # SECTION 7: GEOMETRY CONSISTENCY CHECK
    # -------------------------------------------------------------------
    print("\n--- SECTION 7: GEOMETRY CONSISTENCY CHECK ---")
    gh = f"  {'room_type':<22} {'within_net_boundary':<24} {'overlaps_any_other_room'}"
    print(gh)
    print("  " + "-" * (len(gh) - 2))

    tol = 0.01
    oob_rooms      = 0
    overlap_rooms  = 0

    for r in rooms_list:
        within = (r.x >= -tol and r.y >= -tol and
                  r.x + r.width  <= x_bound + tol and
                  r.y + r.depth  <= y_bound + tol)
        overlaps = any(
            o.room_type != r.room_type and _rooms_overlap(r, o)
            for o in rooms_list
        )
        if not within:
            oob_rooms += 1
        if overlaps:
            overlap_rooms += 1
        wb = "YES" if within  else "NO  <- OUT OF BOUNDS"
        ov = "YES <- OVERLAP" if overlaps else "NO"
        print(f"  {r.room_type:<22} {wb:<24} {ov}")

    print(f"\n  Rooms failing net boundary check:  {oob_rooms}")
    print(f"  Rooms overlapping another room:    {overlap_rooms}")

    return {
        "label":             label,
        "rooms":             len(rooms_list),
        "walls":             len(fp.walls) if has_walls else 0,
        "doors":             len(fp.doors) if has_doors else 0,
        "windows":           len(fp.windows) if has_windows else 0,
        "invalid_geom":      len(invalid_geom),
        "zero_len_walls":    zero_len_walls,
        "oob_walls":         oob_walls,
        "missing_door_pos":  missing_door_pos,
        "missing_swing":     missing_swing,
        "missing_win_pos":   missing_win_pos,
        "missing_win_facing":missing_win_facing,
        "oob_rooms":         oob_rooms,
        "overlap_rooms":     overlap_rooms,
        "missing_fields":    missing_fields,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _contract_ok(stats):
    """True when this floor has no renderer-blocking issues."""
    return (stats["invalid_geom"]       == 0 and
            stats["zero_len_walls"]     == 0 and
            stats["oob_walls"]          == 0 and
            stats["missing_door_pos"]   == 0 and
            stats["missing_swing"]      == 0 and
            stats["missing_win_pos"]    == 0 and
            stats["missing_win_facing"] == 0 and
            stats["oob_rooms"]          == 0 and
            stats["overlap_rooms"]      == 0 and
            len(stats["missing_fields"]) == 0)


def print_summary(all_stats):
    print(f"\n\n{'=' * 92}")
    print("  RENDERER CONTRACT SUMMARY")
    print(f"{'=' * 92}")
    hdr = (f"  {'Case':<34} {'Rooms':>5} {'Walls':>5} {'Doors':>5} {'Wins':>4} "
           f"{'InvGeom':>7} {'0-Wall':>6} {'OOB':>4} "
           f"{'MisDrP':>6} {'MisSwg':>6} {'MisWnP':>6} {'MisWnF':>6} "
           f"{'OOBRm':>5} {'OvlpRm':>6}  Contract")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    all_pass = True
    for s in all_stats:
        ok = _contract_ok(s)
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL <-"
        label = s["label"][:34]
        print(f"  {label:<34} {s['rooms']:>5} {s['walls']:>5} {s['doors']:>5} {s['windows']:>4} "
              f"{s['invalid_geom']:>7} {s['zero_len_walls']:>6} {s['oob_walls']:>4} "
              f"{s['missing_door_pos']:>6} {s['missing_swing']:>6} "
              f"{s['missing_win_pos']:>6} {s['missing_win_facing']:>6} "
              f"{s['oob_rooms']:>5} {s['overlap_rooms']:>6}  {status}")

    print("  " + "-" * (len(hdr) - 2))
    result_line = "ALL FLOORS PASS" if all_pass else "FAILURES DETECTED — see MISSING fields above"
    print(f"\n  {result_line}")
    print(f"{'=' * 92}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all():
    print("=== RENDERER DATA CONTRACT AUDIT ===")
    print(f"  Running {len(CASES)} canonical cases\n")

    all_stats = []

    for pw, pd, bhk, facing, district, floors in CASES:
        base_label = f"{pw}x{pd} {bhk}BHK {facing} {district}"
        result = generate_plan(pw, pd, bhk, facing, district, floors=floors)

        ground_fp = result["ground"]
        label_g   = f"{base_label} G"
        stats_g   = audit_floor(ground_fp, label_g)
        all_stats.append(stats_g)

        if floors >= 2 and "first" in result:
            first_fp = result["first"]
            label_f  = f"{base_label} G+1 (first floor)"
            stats_f  = audit_floor(first_fp, label_f)
            all_stats.append(stats_f)

    print_summary(all_stats)
    print("\n=== END AUDIT ===")


if __name__ == "__main__":
    run_all()
