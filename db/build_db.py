import os
import sqlite3
import sys

import pandas as pd

SEED_DIR = "seeds"
DB_PATH = os.path.join("db", "floorplan.db")
SCHEMA_PATH = os.path.join("db", "schema.sql")

EXPECTED_TABLES = [
    "adjacency_rules",
    "baker_principles",
    "circulation_zones",
    "climate_data",
    "entry_sequence_rules",
    "materials_db",
    "movement_paths",
    "nbc_codes",
    "passage_dimensions",
    "plot_configurations",
    "tn_setbacks",
]


def normalise_columns(columns):
    cleaned = []
    for column in columns:
        value = str(column).strip().lower()
        value = value.replace(" ", "_").replace("-", "_")
        value = (
            value.replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
        )
        cleaned.append(value)
    return cleaned


def safe_index(conn, index_name, table, col):
    cursor = conn.execute(f"PRAGMA table_info({table})")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if col not in existing_cols:
        print(f"  SKIP {index_name} - '{col}' not in {table}")
        return
    conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({col})")
    print(f"  OK  {index_name}")


def main():
    print("SECTION 2 - READ CSV FILES")
    dfs = {}

    if not os.path.isdir(SEED_DIR):
        print(f"  ERR seeds directory not found: {SEED_DIR}")
        sys.exit(1)

    csv_files = sorted(
        [filename for filename in os.listdir(SEED_DIR) if filename.lower().endswith(".csv")]
    )

    for filename in csv_files:
        path = os.path.join(SEED_DIR, filename)
        try:
            try:
                dataframe = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                dataframe = pd.read_csv(path, encoding="latin-1")
            dataframe.columns = normalise_columns(dataframe.columns)
            table_name = os.path.splitext(filename)[0]
            dfs[table_name] = dataframe
            print(f"  OK  {filename}  {len(dataframe)} rows  {len(dataframe.columns)} cols")
        except Exception as error:
            print(f"  ERR {filename}: {error}")

    print(f"\nLoaded tables: {len(dfs)}")

    missing_tables = [table for table in EXPECTED_TABLES if table not in dfs]
    if missing_tables:
        print("Missing expected tables:")
        for table in missing_tables:
            print(f"  - {table}")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != "y":
            print("Aborted by user.")
            sys.exit(1)

    print("\nSECTION 3 - CREATE SQLITE DATABASE")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"  Removed existing DB: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        for table_name, dataframe in dfs.items():
            try:
                dataframe.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"  OK  {table_name}  {len(dataframe)} rows loaded")
            except Exception as error:
                print(f"  ERR {table_name}: {error}")

        print("\nSECTION 4 - CREATE INDEXES")
        safe_index(conn, "idx_adj_room_a", "adjacency_rules", "room_a")
        safe_index(conn, "idx_adj_room_b", "adjacency_rules", "room_b")
        safe_index(conn, "idx_adj_priority", "adjacency_rules", "priority")
        safe_index(conn, "idx_path_from", "movement_paths", "from_space")
        safe_index(conn, "idx_path_to", "movement_paths", "to_space")
        safe_index(conn, "idx_path_critical", "movement_paths", "is_critical_path")
        safe_index(conn, "idx_climate_district", "climate_data", "district")
        safe_index(conn, "idx_nbc_category", "nbc_codes", "code_category")
        safe_index(conn, "idx_nbc_room", "nbc_codes", "room_type")
        safe_index(conn, "idx_mat_category", "materials_db", "material_category")
        safe_index(conn, "idx_mat_baker", "materials_db", "baker_recommended")
        safe_index(conn, "idx_plot_area", "plot_configurations", "plot_area_sqm")
        safe_index(conn, "idx_plot_bhk", "plot_configurations", "bhk_type")
        safe_index(conn, "idx_plot_facing", "plot_configurations", "facing")
        safe_index(conn, "idx_zone_name", "circulation_zones", "zone_name")
        safe_index(conn, "idx_entry_step", "entry_sequence_rules", "sequence_step")
        safe_index(conn, "idx_setback_district", "tn_setbacks", "district")
        safe_index(conn, "idx_setback_area_min", "tn_setbacks", "plot_area_min_sqm")
        safe_index(conn, "idx_passage_type", "passage_dimensions", "passage_type")
        safe_index(conn, "idx_baker_cat", "baker_principles", "category")
        conn.commit()

        print("\nSECTION 5 - EXPORT SCHEMA")
        with open(SCHEMA_PATH, "w", encoding="utf-8") as schema_file:
            for line in conn.iterdump():
                if (
                    line.startswith("CREATE TABLE")
                    or line.startswith("CREATE INDEX")
                    or line.startswith("CREATE UNIQUE")
                ):
                    schema_file.write(line + "\n")
        print(f"  Schema saved to {SCHEMA_PATH}")

        print("\nSECTION 6 - FINAL SUMMARY")
        db_abs_path = os.path.abspath(DB_PATH)
        db_size_kb = os.path.getsize(DB_PATH) / 1024.0
        total_rows = 0
        for table_name in dfs:
            total_rows += conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        print(f"  DB path: {db_abs_path}")
        print(f"  DB size: {db_size_kb:.2f} KB")
        print(f"  Tables loaded: {len(dfs)}")
        print(f"  Total rows: {total_rows}")
        print("Next: python db\\validate_db.py")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
