"""Replay a dataset CSV into a growing 'live' CSV to demo the SCADA dashboard's
Live Plant Feed mode. In a real deployment this file would be written by an
OPC-UA/Modbus gateway or a historian export instead.

Usage:
    python simulate_live_feed.py --source Datasets/swat_test_preprocessed.csv \
        --out live/plant_feed.csv --rate 5 --start-row 251000

Then in the dashboard sidebar: Data Source -> Live Plant Feed (online) ->
Streaming CSV -> path live/plant_feed.csv -> Start Monitoring.
"""
import argparse
import csv
import os
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="Dataset CSV to replay")
    ap.add_argument("--out", default="live/plant_feed.csv", help="Growing 'live' CSV to append to")
    ap.add_argument("--rate", type=float, default=1.0, help="Rows per second (plant sample rate)")
    ap.add_argument("--start-row", type=int, default=0, help="Data row to start replaying from")
    ap.add_argument("--max-rows", type=int, default=0, help="Stop after this many rows (0 = until EOF)")
    ap.add_argument("--keep-label", action="store_true",
                    help="Keep the ground-truth label column (dropped by default — real plants don't label their data)")
    args = ap.parse_args()

    label_cols = {"Normal/Attack", "Label", "Is_Attack", "attack"}

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.source, "r", newline="", encoding="utf-8") as src:
        reader = csv.reader(src)
        header = next(reader)
        header = [h.strip() for h in header]
        if args.keep_label:
            keep_idx = list(range(len(header)))
        else:
            keep_idx = [i for i, h in enumerate(header) if h not in label_cols]

        with open(args.out, "w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow([header[i] for i in keep_idx])
            out.flush()

            n_written = 0
            period = 1.0 / max(args.rate, 1e-6)
            for row_num, row in enumerate(reader):
                if row_num < args.start_row:
                    continue
                writer.writerow([row[i] for i in keep_idx])
                out.flush()
                n_written += 1
                if n_written % max(int(args.rate), 1) == 0:
                    print(f"\r[feed] {n_written} rows sent (source row {row_num})", end="", flush=True)
                if args.max_rows and n_written >= args.max_rows:
                    break
                time.sleep(period)

    print(f"\n[feed] done — {n_written} rows written to {args.out}")


if __name__ == "__main__":
    main()
