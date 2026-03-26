# wire_and_patch.py
# Fixes NBC name mismatch and wires correct DB rules
# into generate_training_data.py constants section

import sqlite3

conn = sqlite3.connect('db/floorplan.db')

# ── Build NBC_MIN_AREA with name normalisation ────────────────────
# Map DB room names → code room names
NBC_NAME_MAP = {
    'bathroom_attached': 'toilet_attached',
    'bathroom_common':   'toilet_common',
    'living_room':       'living',
    'bedroom':           'bedroom_2',   # generic bedroom = secondary bedroom
    'wc':                None,          # skip, too small unit
}

raw_area = conn.execute("""
    SELECT LOWER(room_type), MIN(min_value)
    FROM nbc_codes
    WHERE code_category = 'ROOM_DIMENSIONS'
      AND parameter_name LIKE '%area%'
      AND min_value IS NOT NULL
      AND enforcement_level = 'MANDATORY'
    GROUP BY LOWER(room_type)
""").fetchall()

NBC_MIN_AREA = {
    'master_bedroom':  9.5,
    'bedroom_2':       7.5,
    'bedroom_3':       7.5,
    'bedroom_4':       7.5,
    'living':          9.5,
    'dining':          5.0,
    'kitchen':         4.5,
    'toilet_attached': 2.5,
    'toilet_common':   2.5,
    'utility':         2.0,
    'verandah':        4.0,
    'pooja':           1.2,
    'store':           2.0,
}

# Override with DB values where name maps correctly
for room, val in raw_area:
    mapped = NBC_NAME_MAP.get(room, room)
    if mapped and val:
        NBC_MIN_AREA[mapped] = round(float(val), 1)
    # Also apply bedroom area to all bedroom types
    if room == 'bedroom' and val:
        for br in ['bedroom_2', 'bedroom_3', 'bedroom_4']:
            NBC_MIN_AREA[br] = round(float(val), 1)
    if room == 'master_bedroom' and val:
        NBC_MIN_AREA['master_bedroom'] = round(float(val), 1)
    if room == 'kitchen' and val:
        NBC_MIN_AREA['kitchen'] = round(float(val), 1)

# ── Build NBC_MIN_WIDTH with name normalisation ───────────────────
raw_width = conn.execute("""
    SELECT LOWER(room_type), MIN(min_value)
    FROM nbc_codes
    WHERE code_category = 'ROOM_DIMENSIONS'
      AND parameter_name LIKE '%width%'
      AND min_value IS NOT NULL
      AND enforcement_level = 'MANDATORY'
    GROUP BY LOWER(room_type)
""").fetchall()

NBC_MIN_WIDTH = {
    'master_bedroom':  2.4,
    'bedroom_2':       2.1,
    'bedroom_3':       2.1,
    'bedroom_4':       2.1,
    'living':          2.4,
    'kitchen':         1.8,
    'toilet_attached': 1.2,
    'toilet_common':   1.2,
    'verandah':        1.5,
    'utility':         1.0,
    'dining':          1.8,
}

for room, val in raw_width:
    mapped = NBC_NAME_MAP.get(room, room)
    if mapped and val:
        NBC_MIN_WIDTH[mapped] = round(float(val), 1)
    if room == 'master_bedroom' and val:
        NBC_MIN_WIDTH['master_bedroom'] = round(float(val), 1)
    if room == 'verandah' and val:
        NBC_MIN_WIDTH['verandah'] = round(float(val), 1)
    if room == 'bedroom' and val:
        for br in ['bedroom_2', 'bedroom_3', 'bedroom_4']:
            NBC_MIN_WIDTH[br] = max(
                NBC_MIN_WIDTH.get(br, 2.1),
                round(float(val), 1)
            )

# ── Build FORBIDDEN_ADJ_PAIRS from DB ────────────────────────────
# Normalise room names in adjacency rules too
ADJ_NAME_MAP = {
    'puja_room': 'pooja',
    'bathroom_attached': 'toilet_attached',
    'bathroom_common':   'toilet_common',
    'living_room':       'living',
}

def norm(name):
    return ADJ_NAME_MAP.get(name.lower().strip(), name.lower().strip())

forbidden_rows = conn.execute("""
    SELECT room_a, room_b
    FROM adjacency_rules
    WHERE relationship IN ('MUST_NOT_BE_ADJACENT','FORBIDDEN_ADJACENT')
      AND priority >= 4
""").fetchall()

FORBIDDEN_ADJ_PAIRS = set()
for a, b in forbidden_rows:
    na, nb = norm(a), norm(b)
    FORBIDDEN_ADJ_PAIRS.add((na, nb))
    FORBIDDEN_ADJ_PAIRS.add((nb, na))

# ── Build MUST_SHARE_WALL_PAIRS from DB ──────────────────────────
must_share_rows = conn.execute("""
    SELECT room_a, room_b
    FROM adjacency_rules
    WHERE relationship = 'MUST_SHARE_WALL'
      AND priority >= 4
""").fetchall()

# IMPORTANT: bedroom_2/3/4 ↔ toilet_attached is wrong Fabricate data
# Only master_bedroom should have toilet_attached
# bedroom_2/3/4 use toilet_common
SKIP_MUST_SHARE = {
    ('bedroom_2', 'toilet_attached'),
    ('toilet_attached', 'bedroom_2'),
    ('bedroom_3', 'toilet_attached'),
    ('toilet_attached', 'bedroom_3'),
    ('bedroom_4', 'toilet_attached'),
    ('toilet_attached', 'bedroom_4'),
}

MUST_SHARE_WALL_PAIRS = set()
for a, b in must_share_rows:
    na, nb = norm(a), norm(b)
    pair = (na, nb)
    pair_rev = (nb, na)
    if pair not in SKIP_MUST_SHARE and pair_rev not in SKIP_MUST_SHARE:
        MUST_SHARE_WALL_PAIRS.add(pair)
        MUST_SHARE_WALL_PAIRS.add(pair_rev)

conn.close()

# ── Print summary ─────────────────────────────────────────────────
print("FINAL NBC_MIN_AREA:")
for k, v in sorted(NBC_MIN_AREA.items()):
    print(f"  {k}: {v}")

print("\nFINAL NBC_MIN_WIDTH:")
for k, v in sorted(NBC_MIN_WIDTH.items()):
    print(f"  {k}: {v}")

print(f"\nFINAL FORBIDDEN_ADJ_PAIRS ({len(FORBIDDEN_ADJ_PAIRS)}):")
shown = set()
for a, b in sorted(FORBIDDEN_ADJ_PAIRS):
    pair = tuple(sorted([a,b]))
    if pair not in shown:
        print(f"  {a} <-> {b}")
        shown.add(pair)

print(f"\nFINAL MUST_SHARE_WALL_PAIRS ({len(MUST_SHARE_WALL_PAIRS)}):")
shown = set()
for a, b in sorted(MUST_SHARE_WALL_PAIRS):
    pair = tuple(sorted([a,b]))
    if pair not in shown:
        print(f"  {a} <-> {b}")
        shown.add(pair)

# ── Patch generate_training_data.py ──────────────────────────────
content = open('generate_training_data.py', encoding='utf-8').read()
lines   = content.split('\n')

# Find the NBC_MIN_AREA block and replace through HARDCODED_DEFAULTS
start_idx = None
end_idx   = None

for i, line in enumerate(lines):
    if 'NBC_MIN_AREA' in line and '=' in line and start_idx is None:
        start_idx = i
    if start_idx and 'HARDCODED_DEFAULTS' in line and '=' in line:
        end_idx = i
        break

if start_idx is None or end_idx is None:
    print(f"\nERROR: Could not find constants block. "
          f"start={start_idx} end={end_idx}")
    print("Searching for NBC_MIN_AREA line:")
    for i, line in enumerate(lines):
        if 'NBC_MIN_AREA' in line:
            print(f"  line {i+1}: {line[:80]}")
else:
    new_constants = f'''NBC_MIN_AREA = {repr(NBC_MIN_AREA)}

NBC_MIN_WIDTH = {repr(NBC_MIN_WIDTH)}

FORBIDDEN_ADJ_PAIRS = {repr(FORBIDDEN_ADJ_PAIRS)}

MUST_SHARE_WALL_PAIRS = {repr(MUST_SHARE_WALL_PAIRS)}

'''
    before = '\n'.join(lines[:start_idx])
    after  = '\n'.join(lines[end_idx:])
    new_content = before + '\n' + new_constants + after
    open('generate_training_data.py', 'w', encoding='utf-8').write(new_content)
    print(f"\nPATCHED: replaced lines {start_idx+1} to {end_idx}")
    print("constants now loaded from DB with name normalisation")
