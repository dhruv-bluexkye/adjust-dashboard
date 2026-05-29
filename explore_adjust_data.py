"""
Adjust Data Explorer
Connects to Azure Blob Storage, lists blobs, fetches samples,
and prints schema + recommended dashboard metrics.
"""

import os
import json
import csv
import gzip
import io
from datetime import datetime
from collections import defaultdict

from azure.storage.blob import ContainerClient

ACCOUNT_NAME = "adjustrawdataexports"
CONTAINER_NAME = "adjust-exports"
SAS_TOKEN = "sp=rcwl&st=2026-05-22T12:51:31Z&se=2040-05-22T21:06:31Z&sv=2026-02-06&sr=c&sig=rC20ODItcJQ90nxpQCgKN%2FabHBKtcqPvfcIhdkPu4nQ%3D"

CONTAINER_URL = f"https://{ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}?{SAS_TOKEN}"


def get_container_client():
    return ContainerClient.from_container_url(CONTAINER_URL)


def list_blobs(client, max_blobs=100):
    """List blobs, grouped by prefix/folder structure."""
    blobs = []
    for blob in client.list_blobs():
        blobs.append(blob)
        if len(blobs) >= max_blobs:
            break
    return blobs


def read_blob_sample(client, blob_name, max_rows=200):
    """Download and parse a blob (csv, csv.gz, json, jsonl)."""
    blob_client = client.get_blob_client(blob_name)
    data = blob_client.download_blob().readall()

    # decompress if gzipped
    if blob_name.endswith(".gz"):
        data = gzip.decompress(data)

    text = data.decode("utf-8", errors="replace")

    ext = blob_name.lower().replace(".gz", "")

    if ext.endswith(".csv") or ext.endswith(".tsv"):
        delimiter = "\t" if ext.endswith(".tsv") else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        rows = []
        for i, row in enumerate(reader):
            rows.append(dict(row))
            if i >= max_rows - 1:
                break
        return rows, "csv"

    if ext.endswith(".jsonl") or ext.endswith(".ndjson"):
        rows = []
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if i >= max_rows - 1:
                break
        return rows, "jsonl"

    if ext.endswith(".json"):
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed[:max_rows], "json"
        return [parsed], "json"

    # fallback: try csv
    try:
        reader = csv.DictReader(io.StringIO(text))
        rows = [dict(r) for i, r in enumerate(reader) if i < max_rows]
        return rows, "csv-fallback"
    except Exception:
        return [], "unknown"


def infer_schema(rows):
    """Return column -> set of sample values (max 5) + inferred type."""
    schema = defaultdict(lambda: {"samples": set(), "non_empty": 0, "type": "string"})
    for row in rows:
        for k, v in row.items():
            v = str(v).strip()
            if v:
                schema[k]["non_empty"] += 1
                schema[k]["samples"].add(v)
    # infer types from samples
    for col, info in schema.items():
        samples = list(info["samples"])[:20]
        if all(s.lstrip("-").isdigit() for s in samples if s):
            info["type"] = "integer"
        elif all(_is_float(s) for s in samples if s):
            info["type"] = "float"
        elif any(_is_datetime(s) for s in samples[:5]):
            info["type"] = "datetime"
    return schema


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_datetime(s):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            datetime.strptime(s[:19], fmt)
            return True
        except ValueError:
            pass
    return False


def suggest_dashboard(all_columns):
    """Map known Adjust column names to dashboard sections."""
    cols = {c.lower() for c in all_columns}

    sections = {}

    # ── Acquisition ──────────────────────────────────────────────────────────
    acq = []
    if any(c in cols for c in ["installs", "install_time", "installed_at"]):
        acq.append("Daily / Hourly Installs trend")
    if "network_name" in cols or "tracker_name" in cols:
        acq.append("Installs by Network / Tracker")
    if "country" in cols or "country_code" in cols:
        acq.append("Installs by Country (map)")
    if "campaign_name" in cols or "campaign" in cols:
        acq.append("Installs by Campaign")
    if "adgroup_name" in cols or "creative_name" in cols:
        acq.append("Installs by AdGroup / Creative")
    if "os_name" in cols or "platform" in cols:
        acq.append("Installs by OS / Platform")
    if acq:
        sections["Acquisition"] = acq

    # ── Retention ────────────────────────────────────────────────────────────
    ret = []
    if any(c in cols for c in ["days_since_install", "cohort_day", "retention_day"]):
        ret.append("Day-N Retention curves (D1/D3/D7/D14/D30)")
        ret.append("Retention heatmap by cohort date")
    if "session_count" in cols or "sessions" in cols:
        ret.append("Sessions per user over time")
    if "time_spent" in cols or "session_length" in cols:
        ret.append("Avg session length trend")
    if ret:
        sections["Retention"] = ret

    # ── Revenue / LTV ────────────────────────────────────────────────────────
    rev = []
    if any(c in cols for c in ["revenue", "revenue_usd", "event_revenue"]):
        rev.append("Daily Revenue trend")
        rev.append("Revenue by Network / Campaign")
        rev.append("ARPU (Revenue / Active Users)")
    if "iap_revenue" in cols:
        rev.append("IAP Revenue breakdown")
    if "ad_revenue" in cols:
        rev.append("Ad Revenue breakdown")
    if "ltv" in cols or "predicted_ltv" in cols:
        rev.append("Predicted LTV by cohort")
    if rev:
        sections["Revenue & LTV"] = rev

    # ── Events / Engagement ──────────────────────────────────────────────────
    eng = []
    if "event_name" in cols or "event_token" in cols:
        eng.append("Top Events by count")
        eng.append("Event funnel (ordered event sequences)")
        eng.append("Events per user per day")
    if "level_achieved" in cols or "level" in cols:
        eng.append("Level progression distribution")
    if "purchase" in cols or "purchase_event" in cols:
        eng.append("Purchase funnel conversion")
    if eng:
        sections["Events & Engagement"] = eng

    # ── Attribution / Fraud ──────────────────────────────────────────────────
    attr = []
    if "attribution_type" in cols or "match_type" in cols:
        attr.append("Attribution type breakdown")
    if "is_organic" in cols:
        attr.append("Organic vs Paid split")
    if "is_fraudulent" in cols or "fraud_rejection_reason" in cols:
        attr.append("Fraud rejection rate by network")
    if attr:
        sections["Attribution & Fraud"] = attr

    # ── Device / Geo ─────────────────────────────────────────────────────────
    device = []
    if "device_type" in cols or "device_name" in cols:
        device.append("Installs by Device Type")
    if "app_version" in cols:
        device.append("Active users by App Version")
    if "language" in cols:
        device.append("Users by Language")
    if device:
        sections["Device & Geo"] = device

    return sections


def run():
    print("=" * 70)
    print("  ADJUST DATA EXPLORER")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    client = get_container_client()

    print("\n[1] Listing blobs (up to 100)…")
    blobs = list_blobs(client, max_blobs=100)
    print(f"    Found {len(blobs)} blobs")

    if not blobs:
        print("    Container appears empty.")
        return

    # Show directory structure
    prefixes = defaultdict(int)
    extensions = defaultdict(int)
    for b in blobs:
        parts = b.name.split("/")
        if len(parts) > 1:
            prefixes["/".join(parts[:-1])] += 1
        ext = b.name.split(".")[-1].lower() if "." in b.name else "no-ext"
        if b.name.endswith(".gz"):
            ext = b.name.split(".")[-2].lower() + ".gz"
        extensions[ext] += 1

    print("\n    Folder structure (top-level prefixes):")
    for p, cnt in sorted(prefixes.items())[:20]:
        print(f"      {p}/  ({cnt} files)")

    print("\n    File extensions:")
    for e, cnt in sorted(extensions.items(), key=lambda x: -x[1]):
        print(f"      .{e}: {cnt}")

    # pick up to 3 blobs to sample (prefer recent / small)
    sample_blobs = sorted(blobs, key=lambda b: b.last_modified, reverse=True)[:5]
    # avoid huge files > 50 MB for sampling
    sample_blobs = [b for b in sample_blobs if b.size < 50 * 1024 * 1024] or sample_blobs[:3]
    sample_blobs = sample_blobs[:3]

    print(f"\n[2] Sampling {len(sample_blobs)} recent blobs…")

    all_columns = set()
    schema_by_blob = {}

    for blob in sample_blobs:
        size_kb = blob.size / 1024
        print(f"\n    >> {blob.name}  ({size_kb:.1f} KB, {blob.last_modified:%Y-%m-%d %H:%M})")
        try:
            rows, fmt = read_blob_sample(client, blob.name)
            print(f"      Format: {fmt},  rows sampled: {len(rows)}")
            if rows:
                schema = infer_schema(rows)
                schema_by_blob[blob.name] = schema
                cols = list(schema.keys())
                all_columns.update(c.lower() for c in cols)
                print(f"      Columns ({len(cols)}): {', '.join(cols[:30])}")
                if len(cols) > 30:
                    print(f"      … and {len(cols) - 30} more")
                # print a sample row
                print("      Sample row:")
                sample = rows[0]
                for k, v in list(sample.items())[:15]:
                    print(f"        {k}: {str(v)[:80]}")
        except Exception as exc:
            print(f"      ERROR reading blob: {exc}")

    # Save raw schema to file
    schema_out = {}
    for blob_name, schema in schema_by_blob.items():
        schema_out[blob_name] = {
            col: {
                "type": info["type"],
                "non_empty_rows": info["non_empty"],
                "sample_values": list(info["samples"])[:5],
            }
            for col, info in schema.items()
        }
    with open("schema_report.json", "w") as f:
        json.dump(schema_out, f, indent=2)
    print("\n    Schema saved to schema_report.json")

    # Dashboard recommendations
    print("\n" + "=" * 70)
    print("  RECOMMENDED DASHBOARD SECTIONS")
    print("=" * 70)
    sections = suggest_dashboard(all_columns)

    if not sections:
        print("\n  Could not auto-detect sections — check schema_report.json")
        print(f"  Detected columns: {sorted(all_columns)}")
    else:
        for section, metrics in sections.items():
            print(f"\n  [{section}]")
            for m in metrics:
                print(f"    • {m}")

    print("\n" + "=" * 70)
    print("  Next step: run  python build_dashboard.py  to generate the dashboard")
    print("=" * 70)


if __name__ == "__main__":
    run()
