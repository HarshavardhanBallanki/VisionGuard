import csv
import hashlib
import os
import re
import zipfile

import cv2
import requests
from ultralytics import YOLO

# ── EasyOCR (only OCR engine) ────────────────────────────────────────────────
try:
    import easyocr
    OCR_READER = easyocr.Reader(["en"], gpu=False)
    OCR_AVAILABLE = True
    print("[OCR] EasyOCR loaded OK")
except ImportError:
    OCR_AVAILABLE = False
    print("[OCR] EasyOCR not available — plate text will not be read")

# ── Model paths ───────────────────────────────────────────────────────────────
HELMET_MODEL_PATH = "model/helmet_model.pt"
PLATE_MODEL_PATH  = "model/plate_model.pt"

# ── Tunable constants ─────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE      = 0.40
PLATE_CONF              = 0.20
PLATE_MULTI_SCALES      = (1.0, 2.0)          # was (1.0,1.75,2.5) — 3 YOLO calls → 2
MIN_HEURISTIC_PLATE_SCORE = 14
MIN_PLATE_LENGTH       = 8   # 9-digit bike plate minimum chars (SS+D+LL+NNN)

def is_valid_indian_plate(plate: str) -> bool:
    """
    Full validation: regex match + known state code check.
    Use this before sending to backend or marking as 'detected'.
    """
    if not plate or len(plate) < MIN_PLATE_LENGTH:
        return False
    return bool(PLATE_REGEX.match(plate)) and plate[:2] in INDIAN_STATE_CODES
FRAME_SKIP              = 5                    # was 3 — process every 5th frame (6fps@30fps)
DEDUP_SECONDS           = 2.0

BACKEND_URL  = "http://localhost:5000/api/violation"   # ← adjust if needed
FINE_AMOUNT  = 1000

# ── Class IDs (helmet model) ──────────────────────────────────────────────────
BIKE_ID      = 0
HELMET_ID    = 1
LISC_ID      = 2
NO_HELMET_ID = 3
RIDER_ID     = 5

# ── Misc ──────────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS       = {"jpg", "jpeg", "png"}
PLATE_SEARCH_PAD_X     = 40
PLATE_SEARCH_PAD_TOP   = 30
PLATE_SEARCH_PAD_BOTTOM = 80
MIN_PLATE_LENGTH       = 4
RIDER_HEAD_RATIO       = 0.45
MATCH_OVERLAP_THRESHOLD = 0.10
SEARCH_CONF_FACTOR     = 0.85
FALLBACK_VEHICLE_CONF  = 0.10

INDIAN_STATE_CODES = {
    # States
    "AP","AR","AS","BR","CG","GA","GJ","HR","HP","JH","KA","KL",
    "MP","MH","ML","MN","MZ","NL","OD","OR","PB","RJ","SK","TN",
    "TS","TR","UK","UP","WB",
    # Union Territories
    "AN","CH","DD","DL","DN","JK","LA","LD","PY",
}

# Indian plate regex — covers:
#   9-digit  bikes : SS + 1-2 dist digits + 1-3 series letters + 3-4 reg digits
#   10-digit cars  : SS + 2 dist digits  + 1-3 series letters + 4 reg digits
#   Old/special    : SS + 2 dist digits  + 1   series letter  + 3-4 reg digits
# Total raw char count: 9 (min) to 11 (max), e.g.:
#   KA01AB1234  (10 chars - car)
#   TN1AB1234   (9 chars  - bike, 1-digit district)  → but padded to KA01...
#   AP09Z1234   (9 chars  - bike, 1-series letter)
#   DL3CAB9999  (10 chars - car special)
PLATE_REGEX = re.compile(
    r"^(?:"
    r"[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}"   # standard 10-digit car  (SS-DD-LLL-NNNN)
    r"|[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{3}"  # 9-digit  (SS-DD-LL-NNN)
    r"|[A-Z]{2}[0-9]{1}[A-Z]{1,3}[0-9]{4}"  # 9-digit  bike 1-dist   (SS-D-LLL-NNNN)
    r"|[A-Z]{2}[0-9]{1}[A-Z]{1,3}[0-9]{3}"  # 8-digit  bike 1-dist   (SS-D-LL-NNN)
    r")$"
)

# Global set — each plate is fined exactly once per run
processed_plates: set = set()


# ═══════════════════════════════════════════════════════════════════════════════
#  OCR — single engine, single entry point
# ═══════════════════════════════════════════════════════════════════════════════

def get_best_plate(ocr_results):
    """
    Select the best plate text from EasyOCR results.

    ocr_results: list of (bbox, text, confidence) tuples (detail=1 output).

    Strategy:
      1. Build candidate set: each individual token + left-to-right stitched pairs
         and triplets (handles plates where OCR splits into 2-3 tokens).
      2. Normalise and validate each candidate against PLATE_REGEX.
      3. Among valid hits, prefer: (a) known state code, (b) longer text, (c) higher conf.
      4. If nothing passes regex, return longest high-conf raw text.

    Returns (plate_text, confidence) — plate_text is "" when nothing read.
    """
    if not ocr_results:
        return "", 0.0

    def clean(t):
        return re.sub(r"[^A-Z0-9]", "", t.strip().upper())

    # Build (cleaned_text, avg_conf) candidates from single tokens + stitched combos
    candidates = []
    n = len(ocr_results)
    for i, (_, text, conf) in enumerate(ocr_results):
        c = clean(text)
        if c:
            candidates.append((c, conf))
        # Stitch with next token
        if i + 1 < n:
            c2 = clean(ocr_results[i + 1][1])
            if c and c2:
                merged = c + c2
                avg_c  = (conf + ocr_results[i + 1][2]) / 2.0
                candidates.append((merged, avg_c))
        # Stitch with next two tokens
        if i + 2 < n:
            c2 = clean(ocr_results[i + 1][1])
            c3 = clean(ocr_results[i + 2][1])
            if c and c2 and c3:
                merged = c + c2 + c3
                avg_c  = (conf + ocr_results[i + 1][2] + ocr_results[i + 2][2]) / 3.0
                candidates.append((merged, avg_c))

    valid = []
    raw_best = ("", 0.0)
    for (txt, conf) in candidates:
        normalized = _normalize_plate(txt)
        if conf > raw_best[1]:
            raw_best = (normalized, conf)
        if PLATE_REGEX.match(normalized):
            has_state = normalized[:2] in INDIAN_STATE_CODES
            valid.append((normalized, conf, has_state))

    if valid:
        # Priority: known state code → longer text → higher confidence
        best = max(valid, key=lambda x: (x[2], len(x[0]), x[1]))
        return best[0], best[1]

    # No regex match — return longest normalized raw text, then highest conf
    all_cleaned = [(clean(t), c) for (_, t, c) in ocr_results if t.strip()]
    if all_cleaned:
        best_raw = max(all_cleaned, key=lambda x: (len(x[0]), x[1]))
        return _normalize_plate(best_raw[0]), best_raw[1]

    return raw_best[0], raw_best[1]


def _normalize_plate(text):
    """
    Fix common OCR confusions in Indian plate strings.

    Indian plate structure: SS DD LLL NNNN  (or SS D LLL NNNN for bikes)
      SS   = 2 state letters         → must be letters
      DD   = 1-2 district digits     → must be digits  (O→0, I→1, S→5, B→8, Z→2, G→6)
      LLL  = 1-3 series letters      → must be letters (0→O, 1→I, 5→S, 8→B, 2→Z, 6→G)
      NNNN = 3-4 registration digits → must be digits  (O→0, I→1, S→5, B→8, Z→2, G→6)

    Supports both 9-digit bike plates (SS+D+LLL+NNNN or SS+DD+LL+NNN)
    and 10-digit car/bike plates (SS+DD+LLL+NNNN).

    Deliberately does NOT map A→4 or G→0 — these cause far more mis-parses than
    they fix (A is the most common series start letter; G→0 loses valid G districts).
    """
    t = re.sub(r"[^A-Z0-9]", "", text.strip().upper())
    if len(t) < 5:
        return t

    # Digit-zone confusions: letter that looks like a digit in the district/number zone.
    # B is excluded — it is the most common series letter; converting B→8 in number
    # zone causes far more mis-reads than it fixes.
    # G is excluded for the same reason (common in series like GA, GB, GX etc.)
    # Z→7: OCR frequently reads the digit '7' as the letter 'Z' (they look very similar).
    # In the number/district zone, Z should map to 7, not 2.
    to_digit  = str.maketrans("OISZ", "0157")  # Z→7 (confused with 7 by OCR)
    # Letter-zone confusions: digit that looks like a letter in the series zone.
    # 8→B added: OCR often reads 'B' as '8' in the series zone.
    to_letter = str.maketrans("01258", "OIZSB")

    # Fix state-code zone: correct digit confusions in first 2 chars
    state_raw = t[:2]
    state_fixed = state_raw.translate(to_letter)
    if state_fixed != state_raw:
        t = state_fixed + t[2:]

    def apply_zones(state, dist, series, number):
        return (state
                + dist.translate(to_digit)
                + series.translate(to_letter)
                + number.translate(to_digit))

    def _try_splits(s, only_dist_len=None):
        """Try all structural splits; return (best_valid, first_attempt)."""
        best_valid   = None
        first_attempt = None
        dist_lens  = (2, 1) if only_dist_len is None else (only_dist_len,)
        # Try ALL combinations, collect all valid splits, then rank
        valid_splits = []
        DIST_DIGIT_CONFUSIONS = frozenset("OIZSG")
        for dist_len in dist_lens:
            for series_len in (1, 2, 3):   # 1 first — most common for bike, avoids digit-grab
                for number_len in (4, 3):
                    total = 2 + dist_len + series_len + number_len
                    if len(s) != total:
                        continue
                    state  = s[0:2]
                    dist   = s[2 : 2 + dist_len]
                    series = s[2 + dist_len : 2 + dist_len + series_len]
                    number = s[2 + dist_len + series_len :]
                    # Guard: reject 2-digit district where pos[1] is a true letter
                    if dist_len == 2 and dist[1].isalpha() and dist[1] not in DIST_DIGIT_CONFUSIONS:
                        if first_attempt is None:
                            first_attempt = apply_zones(state, dist, series, number)
                        continue
                    # Guard: series must start with a letter (after zone correction)
                    series_corrected = series.translate(to_letter)
                    if not series_corrected[0].isalpha():
                        if first_attempt is None:
                            first_attempt = apply_zones(state, dist, series, number)
                        continue
                    # Guard: number zone must be ≥50% digit chars after correction
                    number_corrected = number.translate(to_digit)
                    digit_ratio = sum(1 for c in number_corrected if c.isdigit()) / max(1, len(number_corrected))
                    if digit_ratio < 0.5:
                        if first_attempt is None:
                            first_attempt = apply_zones(state, dist, series, number)
                        continue
                    candidate = apply_zones(state, dist, series, number)
                    if PLATE_REGEX.match(candidate):
                        # Score: prefer known state code, longer series is better for cars,
                        # pure-letter series (no digit substitutions) is more reliable
                        has_state = state in INDIAN_STATE_CODES
                        pure_series = all(c.isalpha() for c in series)
                        # Prefer: known state > 2-digit district > pure series > more digits
                        score = (has_state, dist_len, pure_series, number_len)
                        valid_splits.append((score, candidate))
                    if first_attempt is None:
                        first_attempt = apply_zones(state, dist, series, number)
        if valid_splits:
            valid_splits.sort(key=lambda x: x[0], reverse=True)
            return valid_splits[0][1], first_attempt
        return best_valid, first_attempt

    # Fix: OCR may have dropped a digit from a 2-digit district code.
    # Two common OCR failure modes:
    #   (a) Leading digit dropped  → "GA1B7547" from GA01B7547  (insert '0' before)
    #   (b) Repeated digit dropped → "GA1B7547" from GA11B7547  (insert same digit before)
    # Strategy: try inserting the SAME digit first (catches repeated-digit drop),
    # then try '0' (catches leading-zero drop). Return the first valid result.
    # Only trigger when t[3] is NOT already a digit (i.e. district is truly 1-char).
    if (t[:2] in INDIAN_STATE_CODES
            and len(t) >= 6
            and t[2].isdigit()
            and (len(t) < 4 or not t[3].isdigit())):   # district is 1-char → maybe dropped a digit
        two_digit_only, _ = _try_splits(t, only_dist_len=2)
        if two_digit_only is None:
            existing_digit = t[2]   # the digit OCR did read
            # Priority order: same digit first (repeated drop), then '0' (leading-zero drop)
            for insert_digit in dict.fromkeys([existing_digit, "0"]):
                padded = t[:2] + insert_digit + t[2:]
                valid_padded, _ = _try_splits(padded)
                if valid_padded:
                    return valid_padded

    # Primary attempt
    valid, fallback = _try_splits(t)
    if valid:
        return valid

    return fallback if fallback is not None else t


def _detect_plate_lines(gray):
    """
    Detect whether a plate crop is single-line or two-line BEFORE running OCR.
    Uses a horizontal projection profile (row pixel sums) — pure numpy, <1ms.

    Returns 2 if a clear horizontal gap is found (two-line plate),
    otherwise returns 1 (single-line plate).

    How it works:
      1. Threshold the grayscale crop to binary (Otsu).
      2. Sum each row → projection profile.
      3. Normalise by image width so we get average brightness per row.
      4. A two-line plate has a distinct dark band (gap row) between the two
         text lines.  We look for a contiguous run of low-brightness rows
         in the vertical middle third of the image.
    """
    import numpy as np
    if gray is None or gray.size == 0:
        return 1
    h, w = gray.shape[:2]
    if h < 20:
        return 1
    # Otsu threshold → binary (text=white, background=black or vice-versa)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Horizontal projection: mean brightness per row (0-255)
    proj = binary.mean(axis=1)           # shape (h,)
    # Look for a dark gap only in the middle third (ignore top/bottom margins)
    mid_start = h // 3
    mid_end   = 2 * h // 3
    mid_proj  = proj[mid_start:mid_end]
    # A gap row: mean brightness below 35% of max (mostly background)
    threshold = proj.max() * 0.35
    gap_rows  = (mid_proj < threshold).sum()
    # Need at least 8% of crop height as gap to call it two-line
    if gap_rows >= max(2, int(h * 0.08)):
        return 2
    return 1


def _preprocess_crop(gray, plate_lines=1):
    """
    Prepare a grayscale plate crop for EasyOCR.

    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) instead of
    bilateralFilter — CLAHE is ~10x faster and gives better local contrast
    for license plate characters on varied lighting conditions.

    plate_lines: 1 = single-line plate (taller upscale), 2 = two-line plate.
    """
    h, w = gray.shape[:2]
    # Upscale: small crops need more aggressive scaling for OCR to work
    scale = 4 if max(h, w) < 80 else 3
    resized = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    # CLAHE: fast local contrast enhancement — much faster than bilateralFilter d=9
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    return clahe.apply(resized)


def _group_by_line(results, gap_ratio=0.55):
    """
    Split EasyOCR results into top/bottom text lines by vertical position.
    Returns list-of-lines (each sorted L→R), or None if only one line found.
    """
    if not results:
        return None

    def cy(r):  return (r[0][0][1] + r[0][2][1]) / 2.0
    def bh(r):  return abs(r[0][2][1] - r[0][0][1])
    def cx(r):  return (r[0][0][0] + r[0][2][0]) / 2.0

    sorted_r = sorted(results, key=cy)
    heights  = sorted([bh(r) for r in sorted_r])
    median_h = heights[len(heights) // 2] if heights else 1

    lines = [[sorted_r[0]]]
    for r in sorted_r[1:]:
        if cy(r) - cy(lines[-1][-1]) > gap_ratio * median_h:
            lines.append([])
        lines[-1].append(r)

    if len(lines) < 2:
        return None
    return [sorted(line, key=cx) for line in lines]


def _merge_line(line_results):
    """Join all tokens on one line into a single cleaned string."""
    return "".join(
        re.sub(r"[^A-Z0-9]", "", r[1].strip().upper())
        for r in line_results
    )



# ── OCR result cache: bbox_hash → (plate_text, conf, candidates, expires_frame) ──
_ocr_cache: dict = {}
_ocr_cache_ttl_frames: int = 0   # set from fps in run_detection


def _ocr_cache_key(crop):
    """Fast hash of crop dimensions + corner pixels — unique enough for caching."""
    import numpy as np
    h, w = crop.shape[:2]
    # Sample 8 corner/centre pixels for a cheap fingerprint
    samples = [
        int(crop[0, 0].mean()), int(crop[0, w//2].mean()), int(crop[0, -1].mean()),
        int(crop[h//2, 0].mean()), int(crop[h//2, -1].mean()),
        int(crop[-1, 0].mean()), int(crop[-1, w//2].mean()), int(crop[-1, -1].mean()),
        h, w,
    ]
    return hash(tuple(samples))


def ocr_plate_crop(crop, frame_index=0):
    """
    Run EasyOCR on a plate crop.

    Improvements over original:
      1. Pre-detects single-line vs two-line from IMAGE GEOMETRY before OCR
         (horizontal projection profile, <1ms) so the right path runs first time.
      2. Cache: if this exact crop was OCR'd recently, return cached result
         instantly — avoids re-reading the same plate 30× while bike is in frame.
      3. Early exit: if first OCR pass produces a regex-valid plate, return
         immediately without any retry (was always doing up to 2 OCR passes).
      4. Sharp retry only when combined text is too short (<6 chars), meaning
         OCR truly failed — not just when regex didn't match (normalize fixes that).

    Returns (plate_text, ocr_confidence, [candidates])
    """
    if not OCR_AVAILABLE or crop is None or crop.size == 0:
        return "N/A", 0.0, []

    # ── Cache lookup ──────────────────────────────────────────────────────────
    cache_key = _ocr_cache_key(crop)
    if cache_key in _ocr_cache:
        text, conf, cands, expires = _ocr_cache[cache_key]
        if frame_index <= expires:
            return text, conf, cands   # cache hit — free!

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # ── Step 1: detect plate type from image geometry BEFORE running OCR ──────
    num_lines = _detect_plate_lines(gray)
    filtered  = _preprocess_crop(gray, plate_lines=num_lines)

    def _run_ocr(image):
        return OCR_READER.readtext(
            image,
            detail=1,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
        )

    def _cache_and_return(text, conf, cands):
        ttl = max(_ocr_cache_ttl_frames, 15)
        _ocr_cache[cache_key] = (text, conf, cands, frame_index + ttl)
        # Keep cache bounded
        if len(_ocr_cache) > 200:
            oldest = min(_ocr_cache, key=lambda k: _ocr_cache[k][3])
            del _ocr_cache[oldest]
        return text, conf, cands

    try:
        results = _run_ocr(filtered)
    except Exception:
        return "N/A", 0.0, []

    # ── Step 2: route by detected plate type ──────────────────────────────────
    if num_lines == 1:
        # ── SINGLE-LINE PATH ─────────────────────────────────────────────────
        plate_text, conf = get_best_plate(results)
        if not plate_text:
            return _cache_and_return("N/A", 0.0, [])
        # Always normalize — even regex-valid plates may have ambiguous district
        # (e.g. GA1B7547 matches 1-digit district pattern but real plate is GA11B7547)
        plate_text = _normalize_plate(plate_text)
        candidates = [_normalize_plate(re.sub(r"[^A-Z0-9]", "", t.strip().upper()))
                      for _, t, _ in results if t.strip()]
        return _cache_and_return(plate_text, round(conf, 4), candidates)

    # ── TWO-LINE PATH ─────────────────────────────────────────────────────────
    lines = _group_by_line(results)

    if lines is None or len(lines) < 2:
        # OCR didn't split into two lines even though image showed a gap —
        # treat as single line
        plate_text, conf = get_best_plate(results)
        if not plate_text:
            return _cache_and_return("N/A", 0.0, [])
        plate_text = _normalize_plate(plate_text)
        candidates = [_normalize_plate(re.sub(r"[^A-Z0-9]", "", t.strip().upper()))
                      for _, t, _ in results if t.strip()]
        return _cache_and_return(plate_text, round(conf, 4), candidates)

    top_text    = _merge_line(lines[0])
    bottom_text = _merge_line(lines[1])
    combined    = _normalize_plate(top_text + bottom_text)

    def best_conf(line_res):
        return max((r[2] for r in line_res), default=0.0)

    conf = round((best_conf(lines[0]) + best_conf(lines[1])) / 2.0, 4)
    candidates = [_normalize_plate(re.sub(r"[^A-Z0-9]", "", t.strip().upper()))
                  for _, t, _ in results if t.strip()]

    print(f"[OCR] Two-line: top='{top_text}' bottom='{bottom_text}' → '{combined}'")

    # ── Early exit: if normalize produced a valid plate, done — no retry ──────
    if PLATE_REGEX.match(combined):
        return _cache_and_return(combined, conf, candidates)

    # ── Sharp retry ONLY when text is too short (OCR truly missed characters) ──
    # If we got ≥6 chars, _normalize_plate() handles confusion fixes — no 2nd OCR.
    raw_len = len(re.sub(r"[^A-Z0-9]", "", top_text + bottom_text))
    if raw_len >= 6:
        # Enough characters read — normalization already did its job
        return _cache_and_return(combined, conf, candidates)

    # Raw text too short (<6 chars) → OCR truly failed → retry with sharpening
    try:
        blurred   = cv2.GaussianBlur(filtered, (0, 0), sigmaX=2)
        sharpened = cv2.addWeighted(filtered, 1.5, blurred, -0.5, 0)
        retry_results = _run_ocr(sharpened)
        retry_lines   = _group_by_line(retry_results)
        if retry_lines and len(retry_lines) >= 2:
            r_top      = _merge_line(retry_lines[0])
            r_bottom   = _merge_line(retry_lines[1])
            r_combined = _normalize_plate(r_top + r_bottom)
            r_conf     = round((best_conf(retry_lines[0]) + best_conf(retry_lines[1])) / 2.0, 4)
            r_cands    = [_normalize_plate(re.sub(r"[^A-Z0-9]", "", t.strip().upper()))
                          for _, t, _ in retry_results if t.strip()]
            print(f"[OCR] Sharp-retry: top='{r_top}' bottom='{r_bottom}' → '{r_combined}'")
            if PLATE_REGEX.match(r_combined):
                return _cache_and_return(r_combined, r_conf, r_cands)
            # Prefer whichever is longer
            if len(re.sub(r"[^A-Z0-9]", "", r_combined)) > raw_len:
                return _cache_and_return(r_combined, r_conf, r_cands)
    except Exception as e:
        print(f"[OCR] Sharp-retry failed: {e}")

    return _cache_and_return(combined, conf, candidates)




# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_model_path(model_path):
    if os.path.isfile(model_path):
        return model_path
    if os.path.isdir(model_path):
        candidates = [
            os.path.join(model_path, "best"),
            os.path.join(model_path, "best.pt"),
            os.path.join(model_path, "weights", "best.pt"),
            os.path.join(model_path, "last.pt"),
            os.path.join(model_path, "plate_model"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
            if os.path.isdir(c):
                return _repack_torch_archive(c)
    raise FileNotFoundError(f"Model not found: '{model_path}'")


def _repack_torch_archive(source_dir):
    normalized  = os.path.normpath(source_dir)
    archive_name = os.path.basename(normalized.rstrip(r"\/"))
    parent_name  = os.path.basename(os.path.dirname(normalized)) or "model"
    uid          = hashlib.sha1(os.path.abspath(normalized).encode()).hexdigest()[:8]
    cache_dir    = os.path.join("model", "_resolved")
    os.makedirs(cache_dir, exist_ok=True)
    output_path  = os.path.join(cache_dir, f"{parent_name}_{archive_name}_{uid}.pt")
    if os.path.exists(output_path):
        return output_path
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as arc:
        for root, _, files in os.walk(source_dir):
            for name in files:
                full = os.path.join(root, name)
                rel  = os.path.relpath(full, source_dir).replace("\\", "/")
                arc.write(full, f"{archive_name}/{rel}")
    return output_path


def load_helmet_model():
    path  = _resolve_model_path(HELMET_MODEL_PATH)
    model = YOLO(path, task="detect")
    print(f"[Model] Helmet model loaded: {path}")
    return model


def load_plate_model():
    if not os.path.exists(PLATE_MODEL_PATH):
        print(f"[Model] WARNING: No plate model at {PLATE_MODEL_PATH}")
        return None
    path  = _resolve_model_path(PLATE_MODEL_PATH)
    model = YOLO(path, task="detect")
    names = getattr(model.model, "names", {}) or {}
    plate_ids = {
        int(k) for k, v in names.items()
        if any(t in str(v).lower() for t in ("plate", "lisc", "license", "licence"))
    }
    if not plate_ids and len(names) == 1:
        plate_ids = {int(next(iter(names)))}
    model.plate_class_ids = plate_ids or None
    print(f"[Model] Plate model loaded: {path}  classes={model.plate_class_ids}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Backend integration
# ═══════════════════════════════════════════════════════════════════════════════

def send_violation(plate_number):
    """POST a violation to the backend — called once per unique plate.
    Returns the backend JSON (owner, fine, new_balance) or {}.
    """
    payload = {"plate": plate_number, "violation": "No Helmet", "fine": FINE_AMOUNT}
    try:
        resp = requests.post(BACKEND_URL, json=payload, timeout=5)
        print(f"[Backend] {plate_number} → status {resp.status_code}")
        if resp.status_code in (200, 201):
            return resp.json()
    except Exception as exc:
        print(f"[Backend] Request failed for {plate_number}: {exc}")
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
#  Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def clamp_box(box, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def expand_box(box, frame_shape, pad_x=0, pad_y=0, pad_top=None, pad_bottom=None):
    x1, y1, x2, y2 = box
    top    = pad_y if pad_top    is None else pad_top
    bottom = pad_y if pad_bottom is None else pad_bottom
    return clamp_box((x1 - pad_x, y1 - top, x2 + pad_x, y2 + bottom), frame_shape)


def crop_region(frame, box):
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def intersection_area(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def intersection_ratio(inner, outer):
    area = box_area(inner)
    return intersection_area(inner, outer) / area if area > 0 else 0.0


def point_in_box(point, box):
    px, py = point
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]


def combine_boxes(boxes, frame_shape, pad_x=0, pad_y=0, pad_top=None, pad_bottom=None):
    valid = [b for b in boxes if b is not None]
    if not valid:
        return None
    x1 = min(b[0] for b in valid)
    y1 = min(b[1] for b in valid)
    x2 = max(b[2] for b in valid)
    y2 = max(b[3] for b in valid)
    return expand_box((x1, y1, x2, y2), frame_shape, pad_x=pad_x, pad_y=pad_y,
                      pad_top=pad_top, pad_bottom=pad_bottom)


def draw_labeled_box(frame, box, label, box_color, label_bg, label_fg,
                     thickness=2, font_scale=0.72):
    x1, y1, x2, y2 = clamp_box(box, frame.shape)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    tt = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, tt)
    px, py = 10, 8
    lx1 = max(0, x1)
    ly2 = max(th + py * 2 + 2, y1)
    ly1 = max(0, ly2 - th - py * 2)
    lx2 = min(frame.shape[1], lx1 + tw + px * 2)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), label_bg, -1)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 0), 1)
    cv2.putText(frame, label, (lx1 + px, ly2 - py), font, font_scale, label_fg, tt, cv2.LINE_AA)


def draw_violation_overlay(frame, plate_text, box):
    """Clean single-line overlay: PlateNumber | No Helmet | Rs.500"""
    label = f"{plate_text} | No Helmet | Rs.{FINE_AMOUNT}"
    x1, y1, x2, _ = clamp_box(box, frame.shape)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thick = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    px, py = 8, 6
    bx1, by1 = x1, max(0, y1 - th - py * 2 - 4)
    bx2, by2 = min(frame.shape[1], x1 + tw + px * 2), y1
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 200), -1)
    cv2.putText(frame, label, (bx1 + px, by2 - py), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_status_banner(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.9, 2)
    x1, y1 = 10, 20
    x2, y2 = x1 + tw + 24, y1 + th + 20
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), -1)
    cv2.putText(frame, text, (x1 + 12, y2 - 10), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)


def draw_plate_callout(frame, plate_box, plate_text):
    px1, py1, px2, py2 = clamp_box(plate_box, frame.shape)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 215, 255), 2)
    display = plate_text if plate_text not in ("", "N/A") else "Not detected"
    font    = cv2.FONT_HERSHEY_SIMPLEX
    px, py, gap = 12, 10, 8
    (tw, th), _ = cv2.getTextSize("Number Plate:", font, 0.62, 2)
    (vw, vh), _ = cv2.getTextSize(display,        font, 0.70, 2)
    bw = max(tw, vw) + px * 2
    bh = th + vh + py * 2 + gap + 6
    cx1 = min(frame.shape[1] - bw - 5, px2 + 12)
    if cx1 <= px1:
        cx1 = max(5, px1 - bw - 12)
    cy1 = min(frame.shape[0] - bh - 5, max(5, py1 - 6))
    cx2, cy2 = cx1 + bw, cy1 + bh
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 200, 255), -1)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 80, 255), 2)
    cv2.putText(frame, "Number Plate:", (cx1 + px, cy1 + py + th), font, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, display, (cx1 + px, cy1 + py + th + gap + vh), font, 0.70, (0, 0, 0), 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  Geometry / detection helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run_model(model, image, conf, classes=None):
    result = model(image, verbose=False, conf=conf)[0]
    detections = []
    for det in result.boxes:
        class_id = int(det.cls[0])
        if classes is not None and class_id not in classes:
            continue
        x1, y1, x2, y2 = [int(v) for v in det.xyxy[0].tolist()]
        detections.append({"class_id": class_id, "conf": float(det.conf[0]), "bbox": (x1, y1, x2, y2)})
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections


def to_absolute_detections(detections, offset_box, frame_shape):
    ox1, oy1 = offset_box[0], offset_box[1]
    return [{**d, "bbox": clamp_box((ox1 + d["bbox"][0], oy1 + d["bbox"][1],
                                     ox1 + d["bbox"][2], oy1 + d["bbox"][3]), frame_shape)}
            for d in detections]


def rescale_detections(detections, scale):
    if abs(scale - 1.0) < 1e-6:
        return detections
    return [{**d, "bbox": tuple(int(round(v / scale)) for v in d["bbox"])} for d in detections]


def merge_boxes_by_overlap(candidates, frame_shape, overlap_thresh=0.35):
    merged = []
    for c in candidates:
        box = clamp_box(c["bbox"], frame_shape)
        idx = None
        for i, m in enumerate(merged):
            ia = intersection_area(m["bbox"], box)
            if ia / max(1, min(box_area(m["bbox"]), box_area(box))) > overlap_thresh:
                ex1, ey1, ex2, ey2 = m["bbox"]
                cx1, cy1, cx2, cy2 = box
                merged[i]["bbox"]  = clamp_box((min(ex1, cx1), min(ey1, cy1),
                                                max(ex2, cx2), max(ey2, cy2)), frame_shape)
                merged[i]["conf"]  = max(m["conf"], c["conf"])
                idx = i
                break
        if idx is None:
            merged.append({"bbox": box, "conf": c["conf"],
                            "class_id": c.get("class_id", -1)})
    merged.sort(key=lambda d: d["conf"], reverse=True)
    return merged


def merge_vehicle_candidates(candidates, frame_shape):
    merged = []
    for c in candidates:
        box = clamp_box(c["bbox"], frame_shape)
        idx = None
        for i, m in enumerate(merged):
            pe = expand_box(m["bbox"], frame_shape, pad_x=25, pad_y=20)
            pc = expand_box(box, frame_shape, pad_x=15, pad_y=15)
            overlap = intersection_area(pe, pc)
            ea, ca  = max(1, box_area(pe)), max(1, box_area(pc))
            ecx, ecy = bbox_center(m["bbox"])
            ccx, ccy = bbox_center(box)
            ew = max(1, m["bbox"][2] - m["bbox"][0])
            eh = max(1, m["bbox"][3] - m["bbox"][1])
            cw = max(1, box[2] - box[0])
            ch = max(1, box[3] - box[1])
            if (overlap / min(ea, ca) > 0.22
                    and abs(ecx - ccx) <= max(ew, cw) * 0.65
                    and abs(ecy - ccy) <= max(eh, ch) * 0.55):
                ex1, ey1, ex2, ey2 = m["bbox"]
                cx1, cy1, cx2, cy2 = box
                merged[i]["bbox"] = clamp_box((min(ex1, cx1), min(ey1, cy1),
                                               max(ex2, cx2), max(ey2, cy2)), frame_shape)
                merged[i]["conf"] = max(m["conf"], c["conf"])
                idx = i
                break
        if idx is None:
            merged.append({"bbox": box, "conf": c["conf"]})
    return merged


def best_match_for_region(candidates, region_box, used_indexes):
    best_idx, best_score = None, 0.0
    for i, det in enumerate(candidates):
        if i in used_indexes:
            continue
        overlap = intersection_ratio(det["bbox"], region_box)
        cx, cy  = bbox_center(det["bbox"])
        if point_in_box((cx, cy), region_box):
            overlap = max(overlap, 0.25)
        if overlap > best_score:
            best_score, best_idx = overlap, i
    if best_idx is None or best_score < MATCH_OVERLAP_THRESHOLD:
        return None
    return best_idx


def rider_head_box(rider_box, frame_shape):
    x1, y1, x2, y2 = rider_box
    return clamp_box((x1, y1, x2, y1 + int((y2 - y1) * RIDER_HEAD_RATIO)), frame_shape)


# ═══════════════════════════════════════════════════════════════════════════════
#  Vehicle / plate detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_vehicles(frame, helmet_model, confidence_threshold):
    vehicles = run_model(helmet_model, frame,
                         conf=min(confidence_threshold, FALLBACK_VEHICLE_CONF),
                         classes={BIKE_ID})
    if vehicles:
        return vehicles
    # Fallback: use rider/helmet anchors to infer vehicle regions
    anchors = run_model(helmet_model, frame,
                        conf=max(0.10, confidence_threshold * SEARCH_CONF_FACTOR),
                        classes={RIDER_ID, NO_HELMET_ID, HELMET_ID})
    candidates = []
    for a in anchors:
        x1, y1, x2, y2 = a["bbox"]
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        if a["class_id"] == NO_HELMET_ID:
            box = (x1 - int(w * 1.9), y1 - int(h * 0.45), x2 + int(w * 1.9), y2 + int(h * 3.4))
        elif a["class_id"] == HELMET_ID:
            box = (x1 - int(w * 1.55), y1 - int(h * 0.35), x2 + int(w * 1.55), y2 + int(h * 3.1))
        else:
            box = (x1 - int(w * 1.0), y1 - int(h * 0.25), x2 + int(w * 1.1), y2 + int(h * 1.5))
        candidates.append({"bbox": box, "conf": a["conf"]})
    return merge_vehicle_candidates(candidates, frame.shape)


def detect_vehicle_contents(frame, helmet_model, vehicle_box, confidence_threshold):
    crop, abs_box = crop_region(frame, vehicle_box)
    if crop.size == 0:
        return {"vehicle_box": abs_box, "detections": []}
    detections = run_model(helmet_model, crop,
                           conf=max(0.10, confidence_threshold * SEARCH_CONF_FACTOR),
                           classes={HELMET_ID, NO_HELMET_ID, LISC_ID, RIDER_ID})
    return {"vehicle_box": abs_box,
            "detections": to_absolute_detections(detections, abs_box, frame.shape)}


def detect_plates_in_region(frame, plate_model, search_box):
    crop, abs_search_box = crop_region(frame, search_box)
    if crop.size == 0:
        return []
    all_dets = []
    for scale in PLATE_MULTI_SCALES:
        scaled = cv2.resize(crop, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC) if scale > 1.0 else crop
        dets = run_model(plate_model, scaled, conf=PLATE_CONF,
                         classes=getattr(plate_model, "plate_class_ids", None))
        all_dets.extend(rescale_detections(dets, scale))
    absolute = to_absolute_detections(all_dets, abs_search_box, frame.shape)
    return merge_boxes_by_overlap(absolute, frame.shape)


def heuristic_plate_regions(frame, search_box):
    crop, abs_search = crop_region(frame, search_box)
    if crop.size == 0:
        return []
    gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    filtered = cv2.GaussianBlur(gray, (5, 5), 0)   # faster than bilateralFilter for Canny input
    edges   = cv2.Canny(filtered, 30, 180)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sh, sw  = crop.shape[:2]
    proposals = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area   = w * h
        aspect = w / max(1, h)
        cy     = y + h / 2.0
        if area < 200 or area > sw * sh * 0.22:
            continue
        # Indian plates: single-line 2.5:1–5:1, two-line (bike HSRP) 1.0:1–3.0:1
        if not 1.0 <= aspect <= 6.0:
            continue
        if w < max(22, int(sw * 0.05)) or h < max(10, int(sh * 0.030)):
            continue
        if cy < sh * 0.40 or cy > sh * 0.98:
            continue
        abs_box = clamp_box((abs_search[0] + x, abs_search[1] + y,
                              abs_search[0] + x + w, abs_search[1] + y + h), frame.shape)
        proposals.append({"bbox": abs_box, "conf": 0.05, "class_id": -1})
    proposals.sort(key=lambda d: box_area(d["bbox"]), reverse=True)
    return merge_boxes_by_overlap(proposals[:12], frame.shape)


def heuristic_plate_ocr_boxes(frame_shape, vehicle_box):
    x1, y1, x2, y2 = vehicle_box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    boxes = [
        (x1 + int(w * 0.35), y1 + int(h * 0.70), x1 + int(w * 0.68), y1 + int(h * 0.92)),
        (x1 + int(w * 0.32), y1 + int(h * 0.68), x1 + int(w * 0.70), y1 + int(h * 0.94)),
        (x1 + int(w * 0.38), y1 + int(h * 0.72), x1 + int(w * 0.64), y1 + int(h * 0.90)),
    ]
    return [clamp_box(b, frame_shape) for b in boxes]


def rank_plate_candidate(det, anchor_box, vehicle_box):
    pcx, pcy = bbox_center(det["bbox"])
    acx, acy = bbox_center(anchor_box)
    vcx, vcy = bbox_center(vehicle_box)
    return (det["conf"],
            -((pcx - acx) ** 2 + (pcy - acy) ** 2) ** 0.5,
            -((pcx - vcx) ** 2 + (pcy - vcy) ** 2) ** 0.5,
            box_area(det["bbox"]))


def _is_bottom_line_only(text):
    """
    True when OCR captured only the bottom line of a two-line plate
    (series + registration number, without the state code prefix).

    A bottom-line-only read typically looks like: AB1234, CD5678, ZR7493
    It does NOT start with a valid Indian state code.
    """
    t = re.sub(r"[^A-Z0-9]", "", text.strip().upper())
    if len(t) < 3:
        return False
    # Already a full valid plate — not bottom-only
    if PLATE_REGEX.match(t):
        return False
    # If it starts with a known state code → not bottom-only (has top line)
    if len(t) >= 2 and t[:2] in INDIAN_STATE_CODES:
        return False
    # Pattern: starts with 1-3 letters then digits → likely series+number only
    return bool(re.match(r"^[A-Z]{1,3}[0-9]", t))


def _retry_with_top_padding(frame, det_bbox, factor=2.5, frame_index=0):
    """Re-crop with extra upward padding to capture the top line, then OCR."""
    px1, py1, px2, py2 = det_bbox
    ph = max(1, py2 - py1)
    pw = max(1, px2 - px1)
    expanded = expand_box(det_bbox, frame.shape,
                          pad_x=max(12, int(pw * 0.15)),
                          pad_top=max(int(ph * factor), 30),
                          pad_bottom=max(8, int(ph * 0.4)))
    crop, box = crop_region(frame, expanded)
    if crop.size == 0:
        return None
    text, conf, cands = ocr_plate_crop(crop, frame_index=frame_index)
    return text, conf, cands, box


def _is_better(new_text, new_conf, old_text, old_conf):
    """Return True if new result is genuinely better than old."""
    new_clean = re.sub(r"[^A-Z0-9]", "", new_text)
    old_clean = re.sub(r"[^A-Z0-9]", "", old_text)
    new_valid  = bool(PLATE_REGEX.match(new_text))
    old_valid  = bool(PLATE_REGEX.match(old_text))
    if new_valid and not old_valid:
        return True
    if not new_valid and old_valid:
        return False
    if len(new_clean) > len(old_clean):
        return True
    return new_conf > old_conf


def get_plate_for_vehicle(frame, plate_model, vehicle_box, anchor_box, fallback_plates,
                          frame_index=0):
    """
    Locate the best number plate for a vehicle and run OCR on it.
    Returns (plate_text, ocr_conf, [candidates], plate_box).

    Optimisations:
      - Passes frame_index to ocr_plate_crop so the cache works correctly.
      - Early exit: stops trying candidates once a regex-valid plate is found.
      - Heuristic fallback reduced to 1 box (was 3) — only tried if truly needed.
    """
    search_box = expand_box(vehicle_box, frame.shape,
                            pad_x=PLATE_SEARCH_PAD_X,
                            pad_top=PLATE_SEARCH_PAD_TOP,
                            pad_bottom=PLATE_SEARCH_PAD_BOTTOM)
    ranked = []
    if plate_model is not None:
        ranked.extend(detect_plates_in_region(frame, plate_model, search_box))
    ranked.extend(heuristic_plate_regions(frame, search_box))
    ranked.extend(fallback_plates)
    if not ranked:
        return "N/A", 0.0, [], None

    ranked.sort(key=lambda d: rank_plate_candidate(d, anchor_box, vehicle_box), reverse=True)

    best_text, best_conf, best_candidates, best_box = "N/A", 0.0, [], None

    for det in ranked[:4]:
        px1, py1, px2, py2 = det["bbox"]
        # Tight padding: keeps the plate dominant in the crop (~75% of width).
        pad_x = max(6, int((px2 - px1) * 0.15))
        pad_y = max(4, int((py2 - py1) * 0.20))
        crop, plate_box = crop_region(frame, expand_box(det["bbox"], frame.shape,
                                                         pad_x=pad_x, pad_y=pad_y))
        if crop.size == 0:
            continue
        text, conf, cands = ocr_plate_crop(crop, frame_index=frame_index)

        # Two-line retry: bottom-only result → expand upward
        if _is_bottom_line_only(text):
            print(f"[OCR] Bottom-only '{text}' — retrying with top padding")
            retry = _retry_with_top_padding(frame, det["bbox"], frame_index=frame_index)
            if retry:
                r_text, r_conf, r_cands, r_box = retry
                if _is_better(r_text, r_conf, text, conf):
                    text, conf, cands, plate_box = r_text, r_conf, r_cands, r_box
                    print(f"[OCR] Retry → '{text}'")

        if _is_better(text, conf, best_text, best_conf):
            best_text, best_conf, best_candidates, best_box = text, conf, cands, plate_box

        # Early exit: valid plate found — no need to try remaining candidates
        if PLATE_REGEX.match(best_text):
            return best_text, best_conf, best_candidates, best_box

    # Heuristic fallback — only if still no valid result, try best 1 box only
    if not PLATE_REGEX.match(best_text):
        hboxes = heuristic_plate_ocr_boxes(frame.shape, vehicle_box)
        for hbox in hboxes[:1]:   # was [:3] — 1 is enough given YOLO + heuristic already ran
            crop, hb = crop_region(frame, hbox)
            if crop.size == 0:
                continue
            text, conf, cands = ocr_plate_crop(crop, frame_index=frame_index)
            if _is_bottom_line_only(text):
                retry = _retry_with_top_padding(frame, hbox, frame_index=frame_index)
                if retry:
                    r_text, r_conf, r_cands, r_box = retry
                    if _is_better(r_text, r_conf, text, conf):
                        text, conf, cands, hb = r_text, r_conf, r_cands, r_box
            if _is_better(text, conf, best_text, best_conf):
                best_text, best_conf, best_candidates, best_box = text, conf, cands, hb

    return best_text, best_conf, best_candidates, best_box


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-vehicle logic
# ═══════════════════════════════════════════════════════════════════════════════

def build_vehicle_summary(frame, plate_model, vehicle_box, riders, helmets, no_helmets, plates,
                          frame_index=0):
    violations   = []
    annotations  = []
    total_riders = 0
    safe_riders  = 0
    no_helmet_riders = 0
    used_helmets    = set()
    used_no_helmets = set()

    # Determine anchor for plate search
    if no_helmets:
        anchor = no_helmets[0]["bbox"]
    elif helmets:
        anchor = helmets[0]["bbox"]
    elif riders:
        anchor = riders[0]["bbox"]
    else:
        anchor = vehicle_box

    # ── Run OCR only when no-helmet detected AND plate exists ────────────────
    plate_text, plate_conf, plate_candidates, plate_box = "N/A", 0.0, [], None
    if no_helmets and plate_model is not None:
        plate_text, plate_conf, plate_candidates, plate_box = \
            get_plate_for_vehicle(frame, plate_model, vehicle_box, anchor, plates,
                                  frame_index=frame_index)

    # ── Map riders to helmet / no-helmet detections ──────────────────────────
    for rider in riders:
        total_riders += 1
        head_box = rider_head_box(rider["bbox"], frame.shape)
        hi  = best_match_for_region(helmets,    head_box, used_helmets)
        nhi = best_match_for_region(no_helmets, head_box, used_no_helmets)
        best_h  = helmets[hi]    if hi  is not None else None
        best_nh = no_helmets[nhi] if nhi is not None else None

        if best_nh and (not best_h or best_nh["conf"] >= best_h["conf"]):
            used_no_helmets.add(nhi)
            no_helmet_riders += 1
            annotations.append({"bbox": best_nh["bbox"], "label": "No Helmet",
                                 "color": (0, 0, 220), "conf": best_nh["conf"]})
            violations.append({"confidence": round(best_nh["conf"], 4),
                                "plate": plate_text, "plate_conf": plate_conf,
                                "plate_candidates": plate_candidates,
                                "bbox": best_nh["bbox"], "vehicle_bbox": vehicle_box})
        elif best_h:
            used_helmets.add(hi)
            safe_riders += 1
            annotations.append({"bbox": best_h["bbox"], "label": "Helmet",
                                 "color": (0, 200, 0), "conf": best_h["conf"]})

    # ── Fallback: no-rider path ───────────────────────────────────────────────
    if not riders and no_helmets and plate_box is not None:
        vx1, vy1, vx2, vy2 = vehicle_box
        vw, vh = max(1, vx2 - vx1), max(1, vy2 - vy1)
        upper  = vy1 + int(vh * 0.6)
        plausible = []
        for det in no_helmets:
            dx1, dy1, dx2, dy2 = det["bbox"]
            dcx, dcy = bbox_center(det["bbox"])
            dw, dh   = max(1, dx2 - dx1), max(1, dy2 - dy1)
            if dw < max(18, int(vw * 0.08)) or dh < max(18, int(vh * 0.10)):
                continue
            if dy2 <= upper and vx1 <= dcx <= vx2 and vy1 <= dcy <= vy2:
                plausible.append(det)
        if plausible:
            def score(d):
                dx1, dy1, dx2, dy2 = d["bbox"]
                dw2, dh2 = max(1, dx2 - dx1), max(1, dy2 - dy1)
                dcx2, dcy2 = bbox_center(d["bbox"])
                vcx2 = (vx1 + vx2) / 2.0
                ideal = vy1 + vh * 0.22
                return (d["conf"]
                        + min(1.0, (dw2 * dh2) / max(1.0, vw * vh * 0.04))
                        - abs(dcx2 - vcx2) / max(1.0, vw)
                        - abs(dcy2 - ideal) / max(1.0, vh))
            best_nh = max(plausible, key=score)
            total_riders += 1
            no_helmet_riders += 1
            annotations.append({"bbox": best_nh["bbox"], "label": "No Helmet",
                                 "color": (0, 0, 220), "conf": best_nh["conf"]})
            violations.append({"confidence": round(best_nh["conf"], 4),
                                "plate": plate_text, "plate_conf": plate_conf,
                                "plate_candidates": plate_candidates,
                                "bbox": best_nh["bbox"], "vehicle_bbox": vehicle_box})

    # ── Display bounding box ──────────────────────────────────────────────────
    focus = ([r["bbox"] for r in riders]
             + [a["bbox"] for a in annotations]
             + ([plate_box] if plate_box and violations else []))
    display_box = combine_boxes(
        focus, frame.shape,
        pad_x=max(20, int((vehicle_box[2] - vehicle_box[0]) * 0.12)),
        pad_top=max(15, int((vehicle_box[3] - vehicle_box[1]) * 0.05)),
        pad_bottom=max(30, int((vehicle_box[3] - vehicle_box[1]) * 0.10)),
    ) or vehicle_box

    return {
        "violations": violations,
        "annotations": annotations,
        "plate_text": plate_text,
        "plate_conf": plate_conf,
        "plate_box": plate_box,
        "display_box": display_box,
        "total_riders": total_riders,
        "safe_riders": safe_riders,
        "no_helmet_riders": no_helmet_riders,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Frame annotation
# ═══════════════════════════════════════════════════════════════════════════════

def annotate_frame(frame, helmet_model, plate_model, confidence_threshold=DEFAULT_CONFIDENCE,
                   frame_index=0):
    global processed_plates

    vehicles      = detect_vehicles(frame, helmet_model, confidence_threshold)
    total_vehicles = len(vehicles)
    if not vehicles:
        vehicles = [{"bbox": (0, 0, frame.shape[1], frame.shape[0]), "conf": 1.0}]

    total_riders    = 0
    total_helmet    = 0
    total_no_helmet = 0
    violations      = []

    for vehicle in vehicles:
        vehicle_box = clamp_box(vehicle["bbox"], frame.shape)
        contents    = detect_vehicle_contents(frame, helmet_model, vehicle_box, confidence_threshold)
        vehicle_box = contents["vehicle_box"]
        dets        = contents["detections"]

        riders    = [d for d in dets if d["class_id"] == RIDER_ID]
        helmets   = [d for d in dets if d["class_id"] == HELMET_ID]
        no_helmets = [d for d in dets if d["class_id"] == NO_HELMET_ID]
        plates    = [d for d in dets if d["class_id"] == LISC_ID]

        summary = build_vehicle_summary(frame, plate_model, vehicle_box,
                                        riders, helmets, no_helmets, plates,
                                        frame_index=frame_index)
        total_riders    += summary["total_riders"]
        total_helmet    += summary["safe_riders"]
        total_no_helmet += summary["no_helmet_riders"]
        violations.extend(summary["violations"])

        should_draw = bool(summary["total_riders"] > 0 or summary["annotations"] or summary["violations"])
        if total_vehicles and should_draw:
            draw_labeled_box(frame, summary["display_box"], "Bike",
                             (0, 255, 0), (0, 255, 0), (0, 0, 0), thickness=3)

        for ann in summary["annotations"]:
            draw_labeled_box(frame, ann["bbox"], ann["label"],
                             ann["color"], ann["color"],
                             (255, 255, 255) if ann["label"] == "No Helmet" else (0, 0, 0),
                             thickness=2, font_scale=0.75)

        # ── Violation overlay + backend call ─────────────────────────────────
        if summary["violations"] and summary["plate_box"] is not None:
            draw_plate_callout(frame, summary["plate_box"], summary["plate_text"])

        for v in summary["violations"]:
            plate = v["plate"]
            draw_violation_overlay(frame, plate, v["bbox"])

            # (violation sent after deduplication in run_detection — not here)

    if violations:
        draw_status_banner(frame, "NO HELMET DETECTION")

    return frame, violations, total_vehicles, total_riders, total_helmet, total_no_helmet


# ═══════════════════════════════════════════════════════════════════════════════
#  Deduplication & reporting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def frame_to_timestamp(frame_num, fps):
    s = int(frame_num / fps)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _best_plate_from_group(group):
    """
    Pick the most reliable plate text from a group of violation dicts.

    Priority:
      1. PLATE_REGEX-valid plates with known Indian state code — highest OCR conf wins.
      2. PLATE_REGEX-valid plates (any state) — highest OCR conf wins.
      3. Longest plate text, then by confidence.
    """
    valid_with_state = [
        (v["plate"], v.get("plate_conf", 0.0))
        for v in group
        if PLATE_REGEX.match(v.get("plate", "")) and v.get("plate", "")[:2] in INDIAN_STATE_CODES
    ]
    if valid_with_state:
        return max(valid_with_state, key=lambda x: x[1])[0]

    valid = [(v["plate"], v.get("plate_conf", 0.0))
             for v in group
             if PLATE_REGEX.match(v.get("plate", ""))]

    if valid:
        return max(valid, key=lambda x: x[1])[0]

    # No regex-valid plate found — pick longest then highest conf
    return max(group, key=lambda x: (len(x.get("plate", "")),
                                      x.get("plate_conf", 0.0)))["plate"]


def deduplicate(raw_violations, fps):
    """Collapse violations that occur within DEDUP_SECONDS of each other."""
    if not raw_violations:
        return []
    dedup = []
    group = [raw_violations[0].copy()]
    for v in raw_violations[1:]:
        if (v["frame"] - group[-1]["frame"]) / fps <= DEDUP_SECONDS:
            group.append(v.copy())
        else:
            rep = max(group, key=lambda x: x["confidence"]).copy()
            rep["plate"] = _best_plate_from_group(group)
            dedup.append(rep)
            group = [v.copy()]
    rep = max(group, key=lambda x: x["confidence"]).copy()
    rep["plate"] = _best_plate_from_group(group)
    dedup.append(rep)
    return dedup


def group_by_minute(violations):
    buckets = {}
    for v in violations:
        key = ":".join(v["timestamp"].split(":")[:2])
        buckets[key] = buckets.get(key, 0) + 1
    return [{"minute": k, "count": c} for k, c in sorted(buckets.items())]


def calc_compliance(total_helmet, total_no_helmet):
    total = total_helmet + total_no_helmet
    return 100.0 if total == 0 else round(total_helmet / total * 100, 1)


def build_violation_rows(frame_num, timestamp, frame_violations):
    return [
        {"frame": frame_num, "timestamp": timestamp, "type": "No Helmet",
         "confidence": v["confidence"], "plate": v.get("plate", "N/A"),
         "plate_conf": v.get("plate_conf", 0.0),
         "bbox": v.get("bbox"), "vehicle_bbox": v.get("vehicle_bbox")}
        for v in frame_violations
    ]


def _write_csv(report_path, violations):
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "timestamp", "type", "confidence", "plate"])
        writer.writeheader()
        writer.writerows([{k: v[k] for k in ["frame", "timestamp", "type", "confidence", "plate"]}
                          for v in violations])


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def _enrich_violations_with_owner(violations):
    """
    Send each unique valid plate to the wallet backend ONCE (after dedup),
    then attach owner_name / fine_amount / fine_status / remaining_balance
    back onto the violation dict for display in results.html.
    """
    global processed_plates
    for v in violations:
        plate = v.get("plate", "")
        if not is_valid_indian_plate(plate) or plate in processed_plates:
            v.setdefault("owner_name", "—")
            v.setdefault("fine_amount", None)
            v.setdefault("fine_status", None)
            v.setdefault("remaining_balance", "—")
            continue
        processed_plates.add(plate)
        data = send_violation(plate)
        v["owner_name"]  = data.get("owner", "—")
        v["fine_amount"] = data.get("fine", FINE_AMOUNT)
        new_bal = data.get("new_balance")
        if new_bal is not None:
            v["fine_status"]       = "paid" if new_bal > 0 else "low_balance"
            v["remaining_balance"] = new_bal
        else:
            v["fine_status"]       = None
            v["remaining_balance"] = "—"


def run_detection(input_path, output_path, report_path, job_state,
                  confidence_threshold=DEFAULT_CONFIDENCE):
    global processed_plates, _ocr_cache, _ocr_cache_ttl_frames
    processed_plates   = set()   # reset for each new job
    _ocr_cache         = {}      # clear OCR cache for each new job

    ext           = input_path.rsplit(".", 1)[1].lower()
    helmet_model  = load_helmet_model()
    plate_model   = load_plate_model()

    # ── Image mode ───────────────────────────────────────────────────────────
    if ext in IMAGE_EXTENSIONS:
        job_state["step"] = 2
        frame = cv2.imread(input_path)
        if frame is None:
            raise RuntimeError(f"Could not read image: {input_path}")
        annotated, frame_viol, vehicles, riders, helmets, no_helmets = annotate_frame(
            frame, helmet_model, plate_model, confidence_threshold, frame_index=1)
        violations = build_violation_rows(1, "00:00:00", frame_viol)
        job_state["step"] = 5
        _write_csv(report_path, violations)
        _enrich_violations_with_owner(violations)
        count = len(violations)
        return {
            "total_frames": 1, "total_vehicles": vehicles, "total_riders": riders,
            "violation_count": count, "violation_rate": 100.0 if count else 0.0,
            "compliance_rate": calc_compliance(helmets, no_helmets),
            "violations": violations, "by_minute": [],
        }

    # ── Video mode ───────────────────────────────────────────────────────────
    job_state["step"] = 1
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # OCR cache TTL: keep results for DEDUP_SECONDS worth of frames
    _ocr_cache_ttl_frames = max(10, int(fps * DEDUP_SECONDS))

    output_path = output_path.rsplit(".", 1)[0] + ".mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    raw_violations  = []
    total_frames    = 0
    total_vehicles  = 0
    total_riders    = 0
    total_helmet    = 0
    total_no_helmet = 0
    frame_index     = 0
    last_annotated  = None

    job_state["step"] = 2
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame_index % FRAME_SKIP != 0:
            out.write(last_annotated if last_annotated is not None else frame)
            continue

        total_frames += 1
        job_state["step"] = 3

        annotated, frame_viol, vehicles, riders, helmets, no_helmets = annotate_frame(
            frame, helmet_model, plate_model, confidence_threshold, frame_index=frame_index)

        last_annotated   = annotated.copy()
        total_vehicles  += vehicles
        total_riders    += riders
        total_helmet    += helmets
        total_no_helmet += no_helmets

        job_state["step"] = 4
        ts = frame_to_timestamp(frame_index, fps)
        raw_violations.extend(build_violation_rows(frame_index, ts, frame_viol))
        out.write(annotated)

    cap.release()
    out.release()
    job_state["step"] = 5

    violations     = deduplicate(raw_violations, fps)
    violation_count = len(violations)
    violation_rate  = round(violation_count / total_frames * 100, 1) if total_frames > 0 else 0.0
    _write_csv(report_path, violations)
    _enrich_violations_with_owner(violations)

    return {
        "total_frames": total_frames,
        "total_vehicles": total_vehicles,
        "total_riders": total_riders,
        "violation_count": violation_count,
        "violation_rate": violation_rate,
        "compliance_rate": calc_compliance(total_helmet, total_no_helmet),
        "violations": violations,
        "by_minute": group_by_minute(violations),
    }