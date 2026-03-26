import os
import sqlite3
import sys

DB_PATH = os.path.join("db", "floorplan.db")

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

CRITICAL_COLUMNS = {
    "adjacency_rules": ["room_a", "room_b", "relationship", "priority"],
    "movement_paths": [
        "from_space",
        "to_space",
        "path_type",
        "is_critical_path",
        "graph_edge_weight",
    ],
    "plot_configurations": [
        "plot_area_sqm",
        "plot_width_m",
        "plot_depth_m",
        "bhk_type",
        "facing",
        "layout_strategy",
    ],
    "climate_data": [
        "district",
        "climate_zone",
        "window_north_score",
        "window_south_score",
        "window_east_score",
        "window_west_score",
        "floor_plan_orientation_rule",
    ],
    "nbc_codes": [
        "code_category",
        "room_type",
        "parameter_name",
        "min_value",
        "compliance_check",
    ],
    "tn_setbacks": [
        "district",
        "plot_area_min_sqm",
        "front_setback_m",
        "rear_setback_m",
        "side_setback_left_m",
        "side_setback_right_m",
    ],
    "passage_dimensions": [
        "passage_type",
        "min_clear_width_m",
        "recommended_width_m",
        "door_type",
        "drawing_symbol",
        "dimension_annotation",
    ],
    "materials_db": [
        "material_name",
        "material_category",
        "baker_recommended",
        "climate_zone_suitability",
        "wall_drawing_color_hex",
        "hatch_pattern",
    ],
    "baker_principles": [
        "principle_name",
        "category",
        "drawing_impact",
        "cost_saving_pct",
    ],
    "circulation_zones": [
        "zone_name",
        "privacy_level",
        "recommended_zone_fraction",
        "drawing_color_hint",
    ],
    "entry_sequence_rules": [
        "sequence_step",
        "space_type",
        "privacy_transition",
        "drawing_annotation",
    ],
}


def get_columns(conn, table_name):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def print_samples(conn, label, query):
    print(f"{label}:")
    rows = conn.execute(query).fetchall()
    if not rows:
        print("  (no rows)")
        return
    for row in rows:
        print(f"  {row}")


def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: DB file not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    errors = []
    warnings = []

    try:
        print("SECTION 2 - CHECK TABLES")
        db_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        for table in EXPECTED_TABLES:
            if table not in db_tables:
                errors.append(f"Missing table: {table}")
                print(f"  MISS  {table}")
                continue

            row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if row_count == 0:
                errors.append(f"Empty table: {table}")
                print(f"  EMPTY {table}")
            elif row_count < 10:
                warnings.append(f"Low row count in {table}: {row_count}")
                print(f"  WARN  {table}: only {row_count} rows")
            else:
                print(f"  OK    {table}: {row_count} rows")

        print("\nSECTION 3 - CHECK CRITICAL COLUMNS")
        for table, columns in CRITICAL_COLUMNS.items():
            if table not in db_tables:
                continue

            actual_columns = get_columns(conn, table)
            total_rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

            for col in columns:
                if col not in actual_columns:
                    errors.append(f"Missing column: {table}.{col}")
                    print(f"  MISS  {table}.{col}")
                    continue

                null_count = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL"
                ).fetchone()[0]

                if total_rows == 0:
                    pct = 100
                else:
                    pct = round((null_count / total_rows) * 100)

                if pct > 70:
                    warnings.append(f"High nulls in {table}.{col}: {pct}%")
                    print(f"  WARN  {table}.{col}: {pct}% null")
                else:
                    print(f"  OK    {table}.{col}: {pct}% null")

        print("\nSECTION 4 - CRITICAL RULE CHECKS")
        forbidden_count = conn.execute(
            "SELECT COUNT(*) FROM movement_paths WHERE path_type = 'FORBIDDEN'"
        ).fetchone()[0]
        if forbidden_count == 0:
            warnings.append("No FORBIDDEN paths found")
            print("  WARN  No FORBIDDEN paths found")
        else:
            print(f"  OK    {forbidden_count} FORBIDDEN paths found")

        critical_path_count = conn.execute(
            "SELECT COUNT(*) FROM movement_paths WHERE is_critical_path = 1"
        ).fetchone()[0]
        if critical_path_count == 0:
            errors.append("No critical paths (is_critical_path=1)")
            print("  MISS  No critical paths (is_critical_path=1)")
        else:
            print(f"  OK    {critical_path_count} critical paths found")

        bhk_values = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT bhk_type FROM plot_configurations"
            ).fetchall()
            if row[0] is not None
        ]
        if len(bhk_values) < 3:
            warnings.append(f"Fewer than 3 BHK types: {bhk_values}")
        print(f"  OK    BHK types: {bhk_values}")

        district_count = conn.execute(
            "SELECT COUNT(DISTINCT district) FROM climate_data"
        ).fetchone()[0]
        if district_count < 10:
            warnings.append(f"Only {district_count} districts in climate_data")
        print(f"  OK    {district_count} districts in climate_data")

        adjacency_pair_count = conn.execute(
            """
            SELECT COUNT(*) FROM adjacency_rules
            WHERE (room_a='master_bedroom' AND room_b='toilet_attached')
               OR (room_a='toilet_attached' AND room_b='master_bedroom')
            """
        ).fetchone()[0]
        if adjacency_pair_count == 0:
            errors.append("No master_bedroom-toilet_attached rule")
            print("  MISS  No master_bedroom-toilet_attached rule")
        else:
            print("  OK    master_bedroom-toilet_attached rule exists")

        print("\nSECTION 5 - SAMPLE DATA DISPLAY")
        print_samples(
            conn,
            "adjacency_rules",
            "SELECT room_a, room_b, relationship, priority FROM adjacency_rules LIMIT 2",
        )
        print_samples(
            conn,
            "movement_paths",
            "SELECT from_space, to_space, path_type, is_critical_path FROM movement_paths LIMIT 2",
        )
        print_samples(
            conn,
            "climate_data",
            "SELECT district, climate_zone, optimal_plot_facing FROM climate_data LIMIT 2",
        )
        print_samples(
            conn,
            "plot_configurations",
            "SELECT plot_width_m, plot_depth_m, bhk_type, facing, layout_strategy FROM plot_configurations LIMIT 2",
        )
        print_samples(
            conn,
            "tn_setbacks",
            "SELECT district, plot_area_min_sqm, front_setback_m, rear_setback_m FROM tn_setbacks LIMIT 2",
        )

        print("\n" + "=" * 75)
        if errors:
            print(
                f"RESULT: ERRORS FOUND ({len(errors)}) - fix before running generate_training_data.py"
            )
            for error in errors:
                print(f"  X  {error}")
            print("Fix: check your CSV files and re-run python db\\build_db.py")
        elif warnings:
            print(f"RESULT: WARNINGS ONLY ({len(warnings)}) - safe to continue")
            for warning in warnings:
                print(f"  !  {warning}")
            print("Next: python generate_training_data.py")
        else:
            print("RESULT: ALL OK")
            print("Next: python generate_training_data.py")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
