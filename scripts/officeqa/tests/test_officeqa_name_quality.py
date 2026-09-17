"""CPU tests for the name-quality filter + probe-cohort selection (stdlib unittest, no network).

Locks the prompt-quality gate: canonical clean concepts pass; each observed OCR-garble class
(blank, footnote contamination, hyphen+space line breaks, bare acronyms, mangled/real-word
substitutions, near-duplicate spellings) is rejected; and cohort selection is diverse, capped,
magnitude-floored, and deterministic.
"""
import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # repo root
from scripts.officeqa import officeqa_name_quality as Q  # noqa: E402


class TestNameQuality(unittest.TestCase):
    def test_canonical_names_pass(self):
        for c in ["Agriculture Department", "War Department", "Interest on the public debt",
                  "Veterans' Administration", "Federal Old-Age and Survivors Insurance Trust Fund",
                  "Atomic Energy Commission", "Estate and gift taxes"]:
            self.assertTrue(Q.is_clean(c), c)

    def test_spelling_and_punctuation_variants_collapse(self):
        # apostrophe / '&'->'and' / comma / case variants all normalize to the same canonical entry
        for c in ["Veterans Administration", "Veteran's Administration",
                  "Health, Education, & Welfare Department", "Health, Education and Welfare Department",
                  "Housing & Urban Development Department", "interest on the public debt"]:
            self.assertTrue(Q.is_clean(c), c)

    def test_blank_rejected(self):
        self.assertEqual(Q.heuristic_reject(""), "blank")
        self.assertEqual(Q.heuristic_reject("   "), "blank")
        self.assertFalse(Q.is_clean(""))

    def test_footnote_contamination_rejected(self):
        for c, why in [("Aid to agriculture 1/31", "digit"), ("Public works 18'", "digit"),
                       ("Income and profits $/", "symbol"), ("Government and relief 2½/", "digit")]:
            self.assertEqual(Q.heuristic_reject(c), why, c)
            self.assertFalse(Q.is_clean(c), c)
        self.assertEqual(Q.heuristic_reject("Some concept ¼"), "fraction/symbol")

    def test_hyphen_space_linebreak_rejected(self):
        for c in ["Agri- cultural Depart-ment", "Export-Import Bank of Washing- ton",
                  "Veterans' Adminis- tration", "Federal Old -Age and Survivors Insurance Trust Fund"]:
            self.assertEqual(Q.heuristic_reject(c), "hyphen-space", c)
            self.assertFalse(Q.is_clean(c), c)

    def test_bare_acronym_leaf_rejected(self):
        for c in ["VPA", "UERRA", "WPA", "Public works, including work relief > VPA"]:
            self.assertEqual(Q.heuristic_reject(c), "bare-acronym", c)
            self.assertFalse(Q.is_clean(c), c)

    def test_realword_and_ocr_garble_rejected_by_allowlist(self):
        # pass the heuristic (real words / no digits) but miss the canonical allowlist
        for c in ["Heavy Department", "Devy Department", "Savy Department", "Very Department",
                  "Veterans' Admire-tretion", "Micellanea", "Seignorage", "Missoula"]:
            self.assertIsNone(Q.heuristic_reject(c), c)      # heuristic alone would let it through
            self.assertFalse(Q.is_clean(c), c)               # allowlist rejects it

    def test_hierarchical_concept_uses_leaf(self):
        self.assertTrue(Q.is_clean("Budget Outlays > Agriculture Department"))
        self.assertFalse(Q.is_clean("Something > Heavy Department"))


def _q(concept, template, answer, qid):
    return {"qid": qid, "template": template, "pool_role": "main", "answer": answer,
            "golden_path": {"concept": concept}}


class TestCohort(unittest.TestCase):
    def _pool(self):
        # 3 clean concepts x 4 magnitudes (fy) + 1 clean qsg concept + garbled ones (excluded)
        pool = []
        i = 0
        for concept in ["War Department", "Navy Department", "Agriculture Department"]:
            for a in (0.1, 2.0, 8.0, 20.0):
                pool.append(_q(concept, "fy_share_change", a, f"q{i:03d}")); i += 1
        for a in (3.0, 12.0):
            pool.append(_q("Atomic Energy Commission", "quarter_share_gap", a, f"q{i:03d}")); i += 1
        for concept in ["Heavy Department", "Veterans' Admire-tretion 1/31", ""]:
            pool.append(_q(concept, "fy_share_change", 5.0, f"q{i:03d}")); i += 1
        return pool

    def test_min_abs_and_cleanliness_filter(self):
        picks = Q.select_cohort(self._pool(), target=100, per_concept=10, balance_templates=False,
                                min_abs=1.0)
        self.assertTrue(all(abs(p["answer"]) >= 1.0 for p in picks))         # magnitude floor
        self.assertTrue(all(Q.is_clean(p["golden_path"]["concept"]) for p in picks))  # garble gone
        concepts = {Q._norm(Q._leaf(p["golden_path"]["concept"])) for p in picks}
        self.assertEqual(len(concepts), 4)             # 3 clean fy concepts + 1 clean qsg concept

    def test_per_concept_cap_and_determinism(self):
        a = Q.select_cohort(self._pool(), target=100, per_concept=2, balance_templates=False, min_abs=1.0)
        b = Q.select_cohort(self._pool(), target=100, per_concept=2, balance_templates=False, min_abs=1.0)
        self.assertEqual([p["qid"] for p in a], [p["qid"] for p in b])        # deterministic
        from collections import Counter
        by_c = Counter(Q._norm(Q._leaf(p["golden_path"]["concept"])) for p in a)
        self.assertTrue(all(v <= 2 for v in by_c.values()))                  # per-concept cap

    def test_target_caps_total(self):
        picks = Q.select_cohort(self._pool(), target=3, per_concept=10, balance_templates=False, min_abs=1.0)
        self.assertEqual(len(picks), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
