import csv
from datetime import datetime
import os

CALLS_LOG = 'calls_log.csv'
TIER1_PLUMBING = 'tier1_plumbing.csv'

# Read tier1_plumbing.csv to get lead IDs (assume first column is lead_id or use index)
def get_lead_ids():
    lead_ids = []
    with open(TIER1_PLUMBING, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Try to find 'lead_id' column, else use first column
        id_idx = 0
        for i, col in enumerate(header):
            if col.lower() == 'lead_id':
                id_idx = i
                break
        for row in reader:
            if row:
                lead_ids.append(row[id_idx])
    return lead_ids

def ensure_log_exists():
    if not os.path.exists(CALLS_LOG):
        with open(CALLS_LOG, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['lead_id', 'call_time', 'outcome'])

def record_call(lead_id, outcome):
    ensure_log_exists()
    with open(CALLS_LOG, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([lead_id, datetime.now().isoformat(), outcome])

if __name__ == "__main__":
    lead_ids = get_lead_ids()
    assert lead_ids, "No lead IDs found in tier1_plumbing.csv!"
    record_call(lead_ids[0], "pending")
    print("Log created")
    # Verify log exists and has at least 1 row (excluding header)
    with open(CALLS_LOG, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
        assert len(rows) > 1, "calls_log.csv does not have any data rows!"
