# wire_db.py
# Reads adjacency_rules and nbc_codes from DB
# and replaces hardcoded dicts in generate_training_data.py

import sqlite3

conn = sqlite3.connect('db/floorplan.db')

# ── 1. Read NBC min areas from nbc_codes ──────────────────────────
nbc_rows = conn.execute("""
    SELECT LOWER(room_type), MIN(min_value)
    FROM nbc_codes
    WHERE code_category = 'ROOM_DIMENSIONS'
      AND parameter_name LIKE '%area%'
      AND min_value IS NOT NULL
      AND enforcement_level = 'MANDATORY'
    GROUP BY LOWER(room_type)
""").fetchall()

nbc_area_dict = {}
for room, val in nbc_rows:
    # Normalise room names to match ROOM_LISTS keys
    key = (room.replace(' ','_')
               .replace('master bedroom','master_bedroom')
               .replace('bedroom ','bedroom_')
               .replace('toilet attached','toilet_attached')
               .replace('toilet common','toilet_common')
               .strip())
    nbc_area_dict[key] = round(float(val), 1)

print("NBC min areas from DB:")
for k, v in sorted(nbc_area_dict.items()):
    print(f"  {k}: {v}")

# ── 2. Read NBC min widths from nbc_codes ─────────────────────────
nbc_width_rows = conn.execute("""
    SELECT LOWER(room_type), MIN(min_value)
    FROM nbc_codes
    WHERE code_category = 'ROOM_DIMENSIONS'
      AND parameter_name LIKE '%width%'
      AND min_value IS NOT NULL
      AND enforcement_level = 'MANDATORY'
    GROUP BY LOWER(room_type)
""").fetchall()

nbc_width_dict = {}
for room, val in nbc_width_rows:
    key = room.replace(' ','_').strip()
    nbc_width_dict[key] = round(float(val), 1)

print("\nNBC min widths from DB:")
for k, v in sorted(nbc_width_dict.items()):
    print(f"  {k}: {v}")

# ── 3. Read forbidden adjacencies from adjacency_rules ────────────
forbidden_rows = conn.execute("""
    SELECT room_a, room_b
    FROM adjacency_rules
    WHERE relationship IN ('MUST_NOT_BE_ADJACENT','FORBIDDEN_ADJACENT')
      AND priority >= 4
""").fetchall()

forbidden_pairs = set()
for a, b in forbidden_rows:
    forbidden_pairs.add((a, b))
    forbidden_pairs.add((b, a))

print(f"\nForbidden adjacency pairs from DB ({len(forbidden_pairs)}):")
for a, b in sorted(forbidden_pairs):
    print(f"  {a} ↔ {b}")

# ── 4. Read must-share-wall pairs ─────────────────────────────────
must_share_rows = conn.execute("""
    SELECT room_a, room_b
    FROM adjacency_rules
    WHERE relationship = 'MUST_SHARE_WALL'
      AND priority >= 4
""").fetchall()

must_share_pairs = set()
for a, b in must_share_rows:
    must_share_pairs.add((a, b))
    must_share_pairs.add((b, a))

print(f"\nMust-share-wall pairs from DB ({len(must_share_pairs)}):")
for a, b in sorted(must_share_pairs):
    print(f"  {a} ↔ {b}")

conn.close()

# ── 5. Generate new constant block for generate_training_data.py ──
new_block = f'''# ── NBC standards loaded from DB ─────────────────────────────────
NBC_MIN_AREA = {nbc_area_dict}

NBC_MIN_WIDTH = {nbc_width_dict}

# ── Adjacency rules loaded from DB ───────────────────────────────
FORBIDDEN_ADJ_PAIRS = {forbidden_pairs}

MUST_SHARE_WALL_PAIRS = {must_share_pairs}
'''

print("\n" + "="*60)
print("NEW CONSTANTS BLOCK (copy into generate_training_data.py):")
print("="*60)
print(new_block)

# ── 6. Check if current NBC_MIN_AREA in file needs updating ───────
content = open('generate_training_data.py', encoding='utf-8').read()
if 'NBC_MIN_AREA' in content:
    print("NBC_MIN_AREA found in generate_training_data.py")
    print("Review the DB values above vs hardcoded values")
    print("If DB has more rooms covered, update the constants")
else:
    print("NBC_MIN_AREA not found in file")
