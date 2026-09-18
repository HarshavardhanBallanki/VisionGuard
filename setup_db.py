"""
setup_db.py
===========
Run this ONCE before starting VisionGuard to initialise the banking database.

    python setup_db.py

Creates bank_simulation.db in the same folder with 30 pre-loaded demo accounts
covering a range of Indian states, balances (high, medium, low), and names.
These accounts are also used by the deterministic auto-assignment system —
any plate not in this list still gets a consistent fake profile via hashing.
"""

import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "bank_simulation.db"


def _now():
    return datetime.now().strftime("%d %b %Y, %I:%M:%S %p")


def _fake_account_number(plate: str) -> str:
    digest = hashlib.md5(plate.encode()).hexdigest()[:10].upper()
    return f"SBIN{digest}"


def _fake_upi(name: str) -> str:
    clean = name.lower().replace(" ", "")[:8]
    return f"{clean}@oksbi"


# ── 30 Demo accounts ───────────────────────────────────────────────────────────
# Format: (plate, name, phone, aadhar_last4, balance)
# Plates follow Indian format: SS DD [A-Z]{1-3} [0-9]{1-4}
# Balance spread: some high (will pay), some medium, some low (pending challan)

DEMO_ACCOUNTS = [
    # Andhra Pradesh
    ("AP31AB1234", "Ravi Kumar",        "98765XXXXX", "4521", 5000.0),
    ("AP07CD5678", "Suresh Naidu",      "91500XXXXX", "8832", 450.0),
    ("AP26R8104",  "Venkat Rao",        "88001XXXXX", "3310", 3200.0),
    ("AP09MN3301", "Lakshmi Devi",      "97700XXXXX", "7741", 1800.0),

    # Telangana
    ("TS09CD5678", "Priya Sharma",      "87654XXXXX", "2209", 3200.0),
    ("TS07KL9900", "Mohammed Aslam",    "90011XXXXX", "5567", 3750.0),
    ("TS11AB2233", "Sunita Verma",      "85552XXXXX", "9981", 200.0),
    ("TS08GH4455", "Ramesh Babu",       "90909XXXXX", "1123", 950.0),

    # Karnataka
    ("KA05EF9012", "Kiran Reddy",       "76543XXXXX", "6634", 800.0),
    ("KA01ST3344", "Ananya Das",        "78800XXXXX", "4490", 2900.0),
    ("KA03ZX7788", "Geeta Rao",         "87123XXXXX", "2256", 1100.0),

    # Maharashtra
    ("MH12ZZ9999", "Amit Joshi",        "99887XXXXX", "8823", 2500.0),
    ("MH04PQ1122", "Sanjay Patel",      "88001XXXXX", "3341", 4200.0),
    ("MH14AB6677", "Kavita Singh",      "88776XXXXX", "7792", 4800.0),
    ("MH20LM8899", "Rahul Gupta",       "99001XXXXX", "5510", 5500.0),

    # Tamil Nadu
    ("TN22XY4321", "Deepa Nair",        "91234XXXXX", "3302", 650.0),
    ("TN07RS6655", "Arjun Menon",       "93300XXXXX", "4488", 6000.0),
    ("TN18QR2200", "Meena Pillai",      "82200XXXXX", "6673", 3100.0),

    # Kerala
    ("KL09TU5566", "Lakshmi Iyer",      "77990XXXXX", "9921", 1500.0),
    ("KL07VW3344", "Vijay Tiwari",      "96611XXXXX", "7754", 450.0),

    # Delhi
    ("DL01AB0011", "Pooja Agarwal",     "84400XXXXX", "2238", 2100.0),
    ("DL08CD9988", "Dinesh Chandra",    "97700XXXXX", "5563", 1800.0),

    # Rajasthan
    ("RJ14KL5544", "Harish Bhatia",     "98100XXXXX", "3312", 3600.0),
    ("RJ09MN2211", "Seema Chouhan",     "87700XXXXX", "8891", 720.0),

    # Uttar Pradesh
    ("UP32TU8877", "Ranjit Singh",      "99200XXXXX", "4423", 4100.0),
    ("UP16AB3322", "Shobha Mishra",     "85100XXXXX", "6678", 280.0),

    # West Bengal
    ("WB02CD6677", "Subhash Ghosh",     "98300XXXXX", "7712", 2700.0),
    ("WB06EF4433", "Priti Banerjee",    "86600XXXXX", "3345", 1300.0),

    # Gujarat
    ("GJ01ST9911", "Nilesh Shah",       "97800XXXXX", "9934", 5200.0),
    ("GJ06UV2288", "Hetal Patel",       "82900XXXXX", "1167", 380.0),
]


def setup():
    if DB_PATH.exists():
        print(f"[DB] Found existing database at {DB_PATH.name}")
        ans = input("  Reset and reseed? This clears all existing data. (y/N): ").strip().lower()
        if ans != "y":
            print("[DB] Keeping existing database. Done.")
            return
        DB_PATH.unlink()
        print("[DB] Old database deleted.")

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
            plate           TEXT PRIMARY KEY,
            owner_name      TEXT NOT NULL,
            phone           TEXT NOT NULL,
            account_number  TEXT NOT NULL,
            upi_id          TEXT NOT NULL,
            aadhar_last4    TEXT NOT NULL,
            balance         REAL NOT NULL DEFAULT 5000.0,
            created_at      TEXT NOT NULL,
            auto_assigned   INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            plate           TEXT NOT NULL,
            owner_name      TEXT NOT NULL,
            amount          REAL NOT NULL,
            type            TEXT NOT NULL,
            reason          TEXT NOT NULL,
            status          TEXT NOT NULL,
            timestamp       TEXT NOT NULL,
            balance_after   REAL NOT NULL,
            notification    TEXT
        );
    """)

    inserted = 0
    for (plate, name, phone, aadhar, balance) in DEMO_ACCOUNTS:
        acc_no = _fake_account_number(plate)
        upi    = _fake_upi(name)
        cur.execute(
            "INSERT OR IGNORE INTO accounts VALUES (?,?,?,?,?,?,?,?,?)",
            (plate, name, phone, acc_no, upi, aadhar, balance, _now(), 0)
        )
        inserted += cur.rowcount

    conn.commit()
    conn.close()

    print(f"[DB] Database created at {DB_PATH.name}")
    print(f"[DB] {inserted} demo accounts seeded.")
    print()
    print("Sample accounts for testing:")
    print(f"  {'Plate':<14} {'Owner':<20} {'Balance':>10}  Expected result")
    print(f"  {'-'*14} {'-'*20} {'-'*10}  {'-'*30}")
    for plate, name, _, _, balance in DEMO_ACCOUNTS[:8]:
        result = "PAID — Rs.1000 deducted" if balance >= 1000 else "LOW BALANCE — challan issued"
        print(f"  {plate:<14} {name:<20} {balance:>9.0f}  {result}")
    print(f"  ... and {len(DEMO_ACCOUNTS) - 8} more accounts.")
    print()
    print("Any plate NOT in this list will be auto-assigned a consistent fake profile.")
    print("Done. Start VisionGuard with:  python app.py")


if __name__ == "__main__":
    setup()
