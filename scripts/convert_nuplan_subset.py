"""Convert a balanced, deterministic subset of nuPlan mini logs to a ScenarioNet database.

Rule (no cherry-picking): the first N log databases, alphabetically, per city.
Run in the nuPlan environment (see docs/nuplan_pilot.md), with NUPLAN_DATA_ROOT / NUPLAN_MAPS_ROOT set:

    python scripts/convert_nuplan_subset.py -d ~/data/nuplan_db --logs-per-city 2
"""
import argparse
import glob
import os
import sqlite3
from collections import defaultdict

from scenarionet.converter.nuplan.utils import convert_nuplan_scenario, get_nuplan_scenarios
from scenarionet.converter.utils import write_to_directory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--database-path", required=True)
    ap.add_argument("--raw-data-path", default=os.path.expandvars("$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini"))
    ap.add_argument("--logs-per-city", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    by_city = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.raw_data_path, "*.db"))):
        con = sqlite3.connect(f)
        by_city[con.execute("select location from log").fetchone()[0]].append(os.path.basename(f)[:-3])
        con.close()
    logs = [l for city in sorted(by_city) for l in by_city[city][: a.logs_per_city]]
    print("converting logs:", *logs, sep="\n  ")
    scenarios = get_nuplan_scenarios(a.raw_data_path, os.getenv("NUPLAN_MAPS_ROOT"), logs=logs)
    print(f"{len(scenarios)} scenarios")
    write_to_directory(convert_func=convert_nuplan_scenario, scenarios=scenarios, output_path=a.database_path,
                       dataset_version="v1.1", dataset_name="nuplan", overwrite=a.overwrite,
                       num_workers=a.num_workers)


if __name__ == "__main__":
    main()
