# TODO: Fix OCR Over-Processing Bug in detector.py

## Plan
- [x] 1. Simplify `normalize_plate_text()` — reorder to `.upper().strip()`
- [x] 2. Disable `canonicalize_plate_text()` — return `normalize_plate_text(text)` only
- [x] 3. Disable `infer_indian_plate_candidate()` — return `normalize_plate_text(text)` only
- [x] 4. Remove `_plate_slot_char()` helper (no longer used)
- [x] 5. Remove `indian_plate_structure_bonus()` (no longer used)
- [x] 6. Simplify `expand_plate_candidates()` — only normalized texts
- [x] 7. Simplify `plate_text_score()` — no structure forcing
- [x] 8. Simplify `select_best_plate_text()` — remove canonical/structure bias, prefer longest valid OCR
- [x] 9. Simplify `is_better_plate_result()` — remove structure comparison, prefer longer text on tie
- [x] 10. Fix `build_vehicle_summary()` — replace score threshold with text emptiness check
- [x] 11. Fix `annotate_frame()` — remove `MIN_HEURISTIC_PLATE_SCORE` gates (2 locations)

