import sys
import numpy as np
import sqlite3
import pandas as pd

sys.path.insert(0, '.')
from generate_training_data import place_rooms, get_room_targets  # noqa: E402

conn = sqlite3.connect('db/floorplan.db')
plot_confs = pd.read_sql('SELECT * FROM plot_configurations', conn)
conn.close()

rng = np.random.default_rng(42)
net_w, net_d = 10.0, 11.5
bhk = 2
targets = get_room_targets(12.0, 15.0, bhk, plot_confs)
placement, err = place_rooms(net_w, net_d, bhk, targets, rng, error_prob=0.0)

print(f'error_type: {err}')
print(f'net area: {net_w} x {net_d}')
print()
print(f'{"ROOM":<22} {"X":>5} {"Y":>5} {"W":>5} {"D":>5} {"AREA":>6} {"RIGHT":>6} {"TOP":>6}')
print('-' * 65)
for room, r in placement.items():
    area = round(r["w"] * r["d"], 2)
    right = round(r["x"] + r["w"], 2)
    top = round(r["y"] + r["d"], 2)
    print(f'{room:<22} {r["x"]:>5} {r["y"]:>5} {r["w"]:>5} {r["d"]:>5} {area:>6} {right:>6} {top:>6}')
