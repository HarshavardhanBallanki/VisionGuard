"""
trigger.py
==========
One-function bridge between detector.py and the banking system.

detector.py calls:
    from trigger import helmet_violation_detected
    result = helmet_violation_detected(plate_string)

Returns a dict with:
    status            → "paid" | "low_balance" | "error"
    owner_name        → name of the owner (or auto-assigned name)
    fine_amount       → Rs. 1000
    remaining_balance / current_balance
    notification      → simulated SMS string
    timestamp
"""

def helmet_violation_detected(plate: str) -> dict:
    """
    Call this from detector.py for every confirmed no-helmet violation.

    Any plate string works — known plates use seeded data,
    unknown plates get a consistent auto-assigned profile.
    Empty or very short strings are rejected gracefully.
    """
    if not plate or len(plate.strip()) < 2:
        return {
            "status":  "error",
            "message": "Invalid or empty plate string.",
            "owner_name": "Unknown",
            "fine_amount": 0,
        }

    try:
        from bank_engine import BankEngine, DB_PATH
        # Ensure tables exist (idempotent — safe to call every time)
        _ensure_tables(DB_PATH)
        engine = BankEngine()
        return engine.deduct_helmet_fine(plate)
    except Exception as exc:
        return {
            "status":     "error",
            "message":    str(exc),
            "owner_name": "Unknown",
            "fine_amount": 0,
        }


def _ensure_tables(db_path):
    """Create tables if they don't exist yet (first run safety net)."""
    import sqlite3
    with sqlite3.connect(db_path) as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS accounts (
                plate           TEXT PRIMARY KEY,
                owner_name      TEXT NOT NULL,
                phone           TEXT NOT NULL,
                account_number  TEXT NOT NULL,
                upi_id          TEXT NOT NULL,
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


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    tests = [
        "AP31AB1234",  # seeded — Ravi Kumar, Rs.5000
        "TN22XY4321",  # seeded — Deepa Nair, Rs.650 (low balance)
        "MH99XX0000",  # unknown → auto-assigned
        "ABCNOISE1",   # OCR noise → auto-assigned consistently
        "ABCNOISE1",   # same noise string → MUST show same owner as above
    ]
    for p in tests:
        r = helmet_violation_detected(p)
        print(f"\nPlate: {p}")
        for k in ["status","owner_name","fine_amount",
                  "remaining_balance","current_balance","timestamp"]:
            if k in r:
                print(f"  {k}: {r[k]}")
