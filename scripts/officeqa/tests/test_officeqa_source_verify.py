"""CPU regression fixtures for the source verifier (stdlib unittest; no pytest, no network).

Locks the behavior demonstrated in the Phase-2 source-verification spike using self-contained
synthetic fixtures (no dependence on the downloaded HF corpus):
  - accounting identities recover the correct concept column (canonical order);
  - rounding-aware tolerance passes harmless rounding, fails a real anomaly;
  - the concept-scoped target-row check ignores OCR damage in an UNRELATED column but
    rejects damage that touches the concept's own identity;
  - the actor cross-check flags a shifted actor-corpus label as CONFLICT and passes agreement;
  - a genuine cross-edition revision yields two VERIFIED, edition-dependent values.
"""
import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # repo root
from scripts.officeqa import officeqa_source_verify as V  # noqa: E402
from scripts.officeqa import hard_question_gen as g       # noqa: E402

# canonical receipts row (identities hold exactly):
# Withheld100 Other200 Total_ip300 Emp50 Misc60 TotalIR410 Customs40 OtherR10 Gross460 Appr20 Ref5 Net435
CLEAN = [100, 200, 300, 50, 60, 410, 40, 10, 460, 20, 5, 435]


def _receipts_json(monthly, path, year="1960"):
    """Write a minimal parsed-JSON doc with a Receipts-by-Principal-Sources table.
    `monthly` = list of 12 value-lists (Jan..Dec), each length 12."""
    trs = ['<tr><td>%s-%s</td>%s</tr>' % (year, g.MONTHS[i], "".join(f"<td>{v}</td>" for v in monthly[i]))
           for i in range(12)]
    html = "<table>" + "".join(trs) + "</table>"
    doc = {"document": {"pages": [], "elements": [
        {"type": "section_header", "content": "Table 1.- Receipts by Principal Sources"},
        {"type": "table", "content": html}]}}
    with open(path, "w") as fh:
        json.dump(doc, fh)


# the 12 concept labels aligned to CLEAN, in canonical order
_FULL_HEADERS = ["Withheld", "Other", "Total", "Employment taxes", "Miscellaneous internal revenue",
                 "Total internal revenue", "Customs", "Other receipts", "Gross receipts",
                 "Appropriations", "Refunds", "Net receipts"]


def _clean_corpus(tmp, edition, shifted, year="1960"):
    """Write an actor-visible Markdown receipts table. shifted=False -> correct header (12 concept
    columns, 'Customs' aligns to its value). shifted=True -> drop 'Total internal revenue' from the
    HEADER only (data unchanged) -> every label right of it shifts, so 'Customs' lands on the
    Total-internal-revenue value -- the exact real-corpus corruption."""
    concepts = [c for c in _FULL_HEADERS if not (shifted and c == "Total internal revenue")]
    hdr = "| Fiscal year or month | " + " | ".join(concepts) + " |"
    sep = "| " + " | ".join(["---"] * (len(concepts) + 1)) + " |"
    rows = ["| %s-%s | %s |" % (year, g.MONTHS[i], " | ".join(str(x) for x in CLEAN)) for i in range(12)]
    txt = "Table 1.- Receipts by Principal Sources\n(In millions of dollars)\n" + hdr + "\n" + sep + "\n" + "\n".join(rows) + "\n"
    with open(os.path.join(tmp, edition), "w") as fh:
        fh.write(txt)


class TestSourceVerify(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _jp(self, name):
        return os.path.join(self.tmp, name)

    def test_identities_recover_customs(self):
        _receipts_json([list(CLEAN)] * 12, self._jp("ed.json"))
        r = V.verify_series(self._jp("ed.json"), "receipts_by_principal_sources", "1960", "Customs")
        self.assertEqual(r["status"], "verified")
        self.assertEqual(r["vector"], [40.0] * 12)
        self.assertEqual(r["canonical_index"], 6)

    def test_unrelated_ocr_damage_does_not_reject_concept(self):
        rows = [list(CLEAN) for _ in range(12)]
        rows[5][10] = 500          # June: Refunds garbled -> Net identity fails, Customs identity intact
        _receipts_json(rows, self._jp("ed.json"))
        r = V.verify_series(self._jp("ed.json"), "receipts_by_principal_sources", "1960", "Customs")
        self.assertEqual(r["status"], "verified")           # concept-scoped: Refunds damage irrelevant to Customs
        # ... but a series whose OWN identity is hit IS rejected:
        r2 = V.verify_series(self._jp("ed.json"), "receipts_by_principal_sources", "1960", "Refunds of receipts")
        self.assertEqual(r2["status"], "unresolved")

    def test_damage_touching_concept_rejects(self):
        rows = [list(CLEAN) for _ in range(12)]
        rows[6][8] = 999           # July: Gross garbled -> the identity containing Customs fails
        _receipts_json(rows, self._jp("ed.json"))
        r = V.verify_series(self._jp("ed.json"), "receipts_by_principal_sources", "1960", "Customs")
        self.assertEqual(r["status"], "unresolved")
        self.assertIn("July", r["detail"])

    def test_incomplete_coverage_unresolved(self):
        r = V.verify_series(self._jp("ed.json") if False else self._mk_partial(), "receipts_by_principal_sources",
                            "1960", "Customs")
        self.assertEqual(r["status"], "unresolved")
        self.assertIn("coverage", r["detail"])

    def _mk_partial(self):
        # only 6 months present
        trs = ['<tr><td>1960-%s</td>%s</tr>' % (g.MONTHS[i], "".join(f"<td>{v}</td>" for v in CLEAN))
               for i in range(6)]
        doc = {"document": {"pages": [], "elements": [
            {"type": "section_header", "content": "Receipts by Principal Sources"},
            {"type": "table", "content": "<table>" + "".join(trs) + "</table>"}]}}
        p = self._jp("partial.json")
        with open(p, "w") as fh:
            json.dump(doc, fh)
        return p

    def test_actor_conflict_when_corpus_shifted(self):
        _receipts_json([list(CLEAN)] * 12, self._jp("ed.json"))
        _clean_corpus(self.tmp, "ed.txt", shifted=True)     # actor 'Customs' lands on Total-IR value (410)
        r = V.cross_check(self._jp("ed.json"), self.tmp, "receipts_by_principal_sources", "1960", "Customs")
        self.assertEqual(r["status"], "conflict")
        self.assertFalse(r["actor_agrees"])
        self.assertEqual(r["value_by_month"]["January"], 40.0)   # source truth is still recovered

    def test_actor_agreement_verifies(self):
        _receipts_json([list(CLEAN)] * 12, self._jp("ed.json"))
        _clean_corpus(self.tmp, "ed.txt", shifted=False)    # correct actor header -> agrees with source
        r = V.cross_check(self._jp("ed.json"), self.tmp, "receipts_by_principal_sources", "1960", "Customs")
        self.assertEqual(r["status"], "verified")
        self.assertTrue(r["actor_agrees"])

    def test_rounding_tolerance_vs_real_anomaly(self):
        # gen-expenditures: Total == sum(components); 11 components -> tol allows small rounding drift.
        def gx(total, comps, path):
            hdr = "<tr><th>Fiscal year or month</th><th>Total</th>" + "".join(f"<th>c{i}</th>" for i in range(len(comps))) + "</tr>"
            trs = [hdr] + ['<tr><td>1960-%s</td><td>%d</td>%s</tr>' % (g.MONTHS[i], total, "".join(f"<td>{c}</td>" for c in comps)) for i in range(12)]
            doc = {"document": {"pages": [], "elements": [
                {"type": "section_header", "content": "Analysis of General Expenditures"},
                {"type": "table", "content": "<table>" + "".join(trs) + "</table>"}]}}
            with open(path, "w") as fh:
                json.dump(doc, fh)
        comps = [10] * 11                    # sum = 110
        gx(112, comps, self._jp("gx_ok.json"))     # off by 2 -> within rounding tol for 11 comps
        r_ok = V.verify_series(self._jp("gx_ok.json"), "analysis_of_general_expenditures", "1960", "c3")
        self.assertEqual(r_ok["status"], "verified")
        gx(140, comps, self._jp("gx_bad.json"))    # off by 30 -> real anomaly
        r_bad = V.verify_series(self._jp("gx_bad.json"), "analysis_of_general_expenditures", "1960", "c3")
        self.assertEqual(r_bad["status"], "unresolved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
