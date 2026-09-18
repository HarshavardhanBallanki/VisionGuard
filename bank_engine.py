"""
bank_engine.py
==============
Simulated banking system for VisionGuard.

Key behaviours:
  - Every plate (known or unknown, OCR noise included) gets a consistent
    profile via deterministic hashing. Same plate = same owner, every run.
  - If the plate is already in the DB (seeded or previously seen), that
    record is used. Otherwise a profile is auto-created from the pool and
    saved so subsequent calls are consistent.
  - Deducts Rs.1000 on confirmed violation, or marks as pending if balance
    is too low.
  - Simulates an SMS notification on every deduction attempt.
"""

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH     = Path(__file__).parent / "bank_simulation.db"
HELMET_FINE = 1000  # Rs.

# Pool of fake profiles used for auto-assignment.
# Index is picked via MD5(plate) % len(pool) — deterministic across all runs.
FAKE_PROFILE_POOL = [
    {"name": "Ravi Kumar",      "phone": "9876XXXXXX", "balance": 5000.0},
    {"name": "Priya Sharma",    "phone": "8765XXXXXX", "balance": 3200.0},
    {"name": "Kiran Reddy",     "phone": "7654XXXXXX", "balance": 800.0},
    {"name": "Amit Joshi",      "phone": "9988XXXXXX", "balance": 2500.0},
    {"name": "Deepa Nair",      "phone": "9123XXXXXX", "balance": 650.0},
    {"name": "Sanjay Patel",    "phone": "8800XXXXXX", "balance": 4200.0},
    {"name": "Lakshmi Iyer",    "phone": "7799XXXXXX", "balance": 1500.0},
    {"name": "Mohammed Aslam",  "phone": "9001XXXXXX", "balance": 3750.0},
    {"name": "Sunita Verma",    "phone": "8555XXXXXX", "balance": 200.0},
    {"name": "Arjun Menon",     "phone": "9330XXXXXX", "balance": 6000.0},
    {"name": "Geeta Rao",       "phone": "8712XXXXXX", "balance": 1100.0},
    {"name": "Vijay Tiwari",    "phone": "9661XXXXXX", "balance": 450.0},
    {"name": "Ananya Das",      "phone": "7880XXXXXX", "balance": 2900.0},
    {"name": "Rahul Gupta",     "phone": "9900XXXXXX", "balance": 5500.0},
    {"name": "Meena Pillai",    "phone": "8220XXXXXX", "balance": 3100.0},
    {"name": "Suresh Naidu",    "phone": "9150XXXXXX", "balance": 700.0},
    {"name": "Kavita Singh",    "phone": "8877XXXXXX", "balance": 4800.0},
    {"name": "Dinesh Chandra",  "phone": "9770XXXXXX", "balance": 1800.0},
    {"name": "Pooja Agarwal",   "phone": "8440XXXXXX", "balance": 2100.0},
    {"name": "Ramesh Babu",     "phone": "9090XXXXXX", "balance": 950.0},
]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _now():
    return datetime.now().strftime("%d %b %Y, %I:%M:%S %p")

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def _fake_account_number(plate: str) -> str:
    digest = hashlib.md5(plate.encode()).hexdigest()[:10].upper()
    return f"SBIN{digest}"

def _fake_upi(name: str) -> str:
    return f"{name.lower().replace(' ', '')[:8]}@oksbi"

def _assign_profile(plate: str) -> dict:
    """Deterministically pick a fake profile. Same plate → same profile always."""
    idx = int(hashlib.md5(plate.encode()).hexdigest(), 16) % len(FAKE_PROFILE_POOL)
    return FAKE_PROFILE_POOL[idx]

def _ensure_account(plate: str) -> dict:
    """
    Return account dict for plate. Auto-creates if not found.
    Once created, the same profile is returned on all future calls.
    """
    plate = plate.strip().upper()
    with _conn() as c:
        row = c.execute("SELECT * FROM accounts WHERE plate=?", (plate,)).fetchone()
        if row:
            return dict(row)
        # Auto-create from deterministic pool
        profile = _assign_profile(plate)
        acc_no  = _fake_account_number(plate)
        upi     = _fake_upi(profile["name"])
        c.execute(
            "INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?)",
            (plate, profile["name"], profile["phone"],
             acc_no, upi, profile["balance"], _now(), 1)
        )
        print(f"[Bank] Auto-assigned: {plate} → {profile['name']}")
        return {
            "plate": plate, "owner_name": profile["name"],
            "phone": profile["phone"], "account_number": acc_no,
            "upi_id": upi, "balance": profile["balance"], "auto_assigned": 1,
        }

def _simulate_notification(owner: str, phone: str, plate: str,
                            fine: int, balance_after: float, status: str) -> str:
    if status == "paid":
        msg = (
            f"[SMS → {phone}] Dear {owner}, Rs.{fine} has been automatically "
            f"deducted from your linked account for a No Helmet violation on "
            f"vehicle {plate}. Remaining balance: Rs.{balance_after:.2f}. "
            f"— VisionGuard Traffic Enforcement"
        )
    else:
        msg = (
            f"[SMS → {phone}] Dear {owner}, a No Helmet challan of Rs.{fine} "
            f"has been issued for vehicle {plate}. Your balance is insufficient. "
            f"Please pay within 7 days to avoid further action. "
            f"— VisionGuard Traffic Enforcement"
        )
    print(msg)
    return msg


# ── Public API ─────────────────────────────────────────────────────────────────

class BankEngine:

    def get_account(self, plate: str) -> dict:
        """Return account details. Auto-creates if not registered."""
        plate   = plate.strip().upper()
        account = _ensure_account(plate)
        return {
            "status":         "found",
            "plate":          account["plate"],
            "owner_name":     account["owner_name"],
            "phone":          account["phone"],
            "account_number": account["account_number"],
            "upi_id":         account["upi_id"],
            "balance":        round(account["balance"], 2),
            "auto_assigned":  bool(account.get("auto_assigned", 0)),
        }

    def deduct_helmet_fine(self, plate: str) -> dict:
        """
        Main entry point. Deducts Rs.1000 fine for a no-helmet violation.
        Returns a result dict consumed by trigger.py and shown on results page.
        """
        plate   = plate.strip().upper()
        account = _ensure_account(plate)
        owner   = account["owner_name"]
        phone   = account["phone"]
        acc_no  = account["account_number"]
        upi     = account["upi_id"]
        balance = account["balance"]

        with _conn() as c:
            if balance >= HELMET_FINE:
                new_balance = round(balance - HELMET_FINE, 2)
                c.execute("UPDATE accounts SET balance=? WHERE plate=?",
                          (new_balance, plate))
                notif = _simulate_notification(owner, phone, plate,
                                               HELMET_FINE, new_balance, "paid")
                c.execute(
                    "INSERT INTO transactions "
                    "(plate,owner_name,amount,type,reason,status,timestamp,balance_after,notification) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (plate, owner, HELMET_FINE, "debit",
                     "No Helmet Fine – Traffic Challan",
                     "success", _now(), new_balance, notif)
                )
                return {
                    "status":             "paid",
                    "plate":              plate,
                    "owner_name":         owner,
                    "phone":              phone,
                    "account_number":     acc_no,
                    "upi_id":             upi,
                    "fine_amount":        HELMET_FINE,
                    "previous_balance":   round(balance, 2),
                    "remaining_balance":  new_balance,
                    "reason":             "No Helmet Detected – Automatic Challan",
                    "notification":       notif,
                    "timestamp":          _now(),
                }
            else:
                notif = _simulate_notification(owner, phone, plate,
                                               HELMET_FINE, balance, "pending")
                c.execute(
                    "INSERT INTO transactions "
                    "(plate,owner_name,amount,type,reason,status,timestamp,balance_after,notification) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (plate, owner, HELMET_FINE, "debit",
                     "No Helmet Fine – PENDING (low balance)",
                     "pending", _now(), balance, notif)
                )
                return {
                    "status":          "low_balance",
                    "plate":           plate,
                    "owner_name":      owner,
                    "phone":           phone,
                    "account_number":  acc_no,
                    "fine_amount":     HELMET_FINE,
                    "current_balance": round(balance, 2),
                    "shortfall":       round(HELMET_FINE - balance, 2),
                    "message":         "Challan issued. Balance low — penalty notice sent.",
                    "notification":    notif,
                    "timestamp":       _now(),
                }

    def get_transactions(self, plate: str, limit: int = 10) -> list:
        plate = plate.strip().upper()
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM transactions WHERE plate=? ORDER BY id DESC LIMIT ?",
                (plate, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def all_accounts(self) -> list:
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM accounts ORDER BY plate"
            ).fetchall()
            return [dict(r) for r in rows]

    def add_account(self, plate: str, name: str, phone: str,
                    balance: float = 5000.0) -> dict:
        """Manually register a vehicle (overrides auto-assignment)."""
        plate = plate.strip().upper()
        with _conn() as c:
            if c.execute("SELECT plate FROM accounts WHERE plate=?",
                         (plate,)).fetchone():
                return {"status": "exists", "plate": plate}
            c.execute(
                "INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?)",
                (plate, name.strip(), phone.strip(),
                 _fake_account_number(plate), _fake_upi(name),
                 float(balance), _now(), 0)
            )
        return {"status": "created", "plate": plate,
                "name": name, "balance": balance}
