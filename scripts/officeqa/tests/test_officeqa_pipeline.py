"""CPU tests for the generation pipeline: composition math + negative controls (stdlib unittest).

Covers the reviewer's spike asks: independently-checked composition arithmetic, ill-posed rejection,
and planted-defect negative controls (unit error, missing month, wrong year, and a column SWAP that
the accounting identity cannot see but the actor cross-check catches).
"""
import json
import os
import re
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # repo root
from scripts.officeqa import officeqa_pipeline as P        # noqa: E402
from scripts.officeqa import officeqa_source_verify as V    # noqa: E402
from scripts.officeqa.hard_question_gen import MONTHS        # noqa: E402

# Alpha spikes in Q1; Beta,Gamma constant. Total = Alpha+Beta+Gamma.
ALPHA = [40, 40, 40, 10, 10, 10, 10, 10, 10, 10, 10, 10]
BETA = [10] * 12
GAMMA = [10] * 12
TOTAL = [ALPHA[i] + BETA[i] + GAMMA[i] for i in range(12)]
# independent quarter_share_gap: Q1 share 120/180=.6667, annual 210/450=.46667 -> 20.00 pp
EXPECT_GAP = 20.0
CONCEPTS = ["Alpha", "Beta", "Gamma"]


def _mk_vs(vec, tot, concept="Alpha", year="1955"):
    return V.VerifiedSeries(concept, year, "treasury_bulletin_1955_06.json", '"Other" Expenditures',
                            "total_components", list(vec), list(tot), 0, 0, True)


def _rows(swap=None):
    """12 monthly value-lists [Total, Alpha, Beta, Gamma]; swap swaps two data indices per row."""
    out = []
    for i in range(12):
        v = [TOTAL[i], ALPHA[i], BETA[i], GAMMA[i]]
        if swap:
            v[swap[0]], v[swap[1]] = v[swap[1]], v[swap[0]]
        out.append(v)
    return out


def _exp_json(path, rows, year="1955", n_months=12):
    hdr = "<tr><th>Fiscal year or month</th><th>Total</th>" + "".join(f"<th>{c}</th>" for c in CONCEPTS) + "</tr>"
    trs = [hdr] + ['<tr><td>%s-%s</td>%s</tr>' % (year, MONTHS[i], "".join(f"<td>{v}</td>" for v in rows[i]))
                   for i in range(n_months)]
    doc = {"document": {"pages": [], "elements": [
        {"type": "section_header", "content": 'Table 5.- "Other" Expenditures'},
        {"type": "table", "content": "<table>" + "".join(trs) + "</table>"}]}}
    with open(path, "w") as fh:
        json.dump(doc, fh)


def _exp_txt(path, rows, year="1955", n_months=12):
    hdr = "| Fiscal year or month | Total | " + " | ".join(CONCEPTS) + " |"
    sep = "| " + " | ".join(["---"] * (len(CONCEPTS) + 2)) + " |"
    body = ["| %s-%s | %s |" % (year, MONTHS[i], " | ".join(str(v) for v in rows[i])) for i in range(n_months)]
    with open(path, "w") as fh:
        fh.write('Table 5.- "Other" Expenditures\n(In millions of dollars)\n' + hdr + "\n" + sep + "\n" + "\n".join(body) + "\n")


class TestCompositionMath(unittest.TestCase):
    def test_quarter_share_gap_value(self):
        q = P.compose_quarter_share_gap(_mk_vs(ALPHA, TOTAL))
        self.assertIsNotNone(q)
        self.assertEqual(q["answer"], EXPECT_GAP)
        self.assertEqual(q["pool_role"], "main")
        self.assertNotIn("Table", q["question"])          # no location hints
        self.assertIn("percentage points", q["unit"])

    def test_year_sum_control(self):
        q = P.compose_year_sum(_mk_vs(ALPHA, TOTAL))
        self.assertEqual(q["answer"], round(sum(ALPHA), 4))
        self.assertEqual(q["pool_role"], "control")

    def test_ill_posed_rejected(self):
        neg_q = [-40, -40, -40] + [10] * 9                                                # Q1 sum < 0 -> negative share
        self.assertIsNone(P.compose_quarter_share_gap(_mk_vs(neg_q, TOTAL)))
        self.assertIsNone(P.compose_quarter_share_gap(_mk_vs(ALPHA, [0] * 12)))            # zero denominators
        self.assertIsNone(P.compose_quarter_share_gap(_mk_vs([0] * 12, TOTAL)))            # zero component

    def test_qid_deterministic_and_specific(self):
        a = P.compose_quarter_share_gap(_mk_vs(ALPHA, TOTAL))
        b = P.compose_quarter_share_gap(_mk_vs(ALPHA, TOTAL))
        self.assertEqual(a["qid"], b["qid"])                                              # deterministic
        c = P.compose_quarter_share_gap(_mk_vs(ALPHA, TOTAL, year="1956"))
        self.assertNotEqual(a["qid"], c["qid"])                                           # year in the spec


class TestEnumerationAndControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        V._CLEAN_CACHE.clear()

    def _p(self, n):
        return os.path.join(self.tmp, n)

    def test_clean_series_verifies_and_agrees(self):
        _exp_json(self._p("treasury_bulletin_1955_06.json"), _rows())
        _exp_txt(self._p("treasury_bulletin_1955_06.txt"), _rows())
        got = V.enumerate_verified_series(self._p("treasury_bulletin_1955_06.json"), self.tmp)
        alpha = [s for s in got if V._leaf(s.concept) == "alpha" and s.year == "1955"]
        self.assertTrue(alpha and alpha[0].actor_agrees)
        self.assertEqual(alpha[0].vector, ALPHA)
        self.assertEqual(alpha[0].total_vector, TOTAL)

    def test_unit_error_row_rejects_series(self):
        rows = _rows(); rows[5][1] = ALPHA[5] * 1000          # June Alpha blown up -> Total!=sum
        _exp_json(self._p("treasury_bulletin_1955_06.json"), rows)
        r = V.verify_series(self._p("treasury_bulletin_1955_06.json"), "total_components", "1955",
                            "Alpha", caption_re=re.compile("Expenditures", re.I))
        self.assertEqual(r["status"], "unresolved")

    def test_missing_month_incomplete(self):
        _exp_json(self._p("treasury_bulletin_1955_06.json"), _rows(), n_months=11)   # only 11 months
        r = V.verify_series(self._p("treasury_bulletin_1955_06.json"), "total_components", "1955",
                            "Alpha", caption_re=re.compile("Expenditures", re.I))
        self.assertEqual(r["status"], "unresolved")
        self.assertIn("coverage", r["detail"])

    def test_wrong_year_incomplete(self):
        _exp_json(self._p("treasury_bulletin_1955_06.json"), _rows(), year="1955")
        r = V.verify_series(self._p("treasury_bulletin_1955_06.json"), "total_components", "1975",
                            "Alpha", caption_re=re.compile("Expenditures", re.I))
        self.assertEqual(r["status"], "unresolved")

    def test_column_swap_missed_by_identity_caught_by_actor(self):
        # SOURCE swaps Alpha<->Beta (data idx 1<->2): Total=sum still holds (sum is permutation-invariant)
        _exp_json(self._p("treasury_bulletin_1955_06.json"), _rows(swap=(1, 2)))
        # source alone verifies (identity can't see the swap):
        r = V.verify_series(self._p("treasury_bulletin_1955_06.json"), "total_components", "1955",
                            "Alpha", caption_re=re.compile("Expenditures", re.I))
        self.assertEqual(r["status"], "verified")
        self.assertEqual(r["vector"], BETA)                  # returns Beta's values under 'Alpha' (wrong)
        # the CORRECT actor corpus disagrees -> cross_check quarantines it:
        _exp_txt(self._p("treasury_bulletin_1955_06.txt"), _rows())     # correct
        rc = V.cross_check(self._p("treasury_bulletin_1955_06.json"), self.tmp, "total_components",
                           "1955", "Alpha", caption_re=re.compile("Expenditures", re.I))
        self.assertEqual(rc["status"], "conflict")
        self.assertFalse(rc["actor_agrees"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
