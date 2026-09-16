"""CPU unit tests for the hard-question generator (stdlib unittest; no pytest dep).

Run: python3 -m unittest scripts.officeqa.tests.test_hard_question_gen -v
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from scripts.officeqa import hard_question_gen as g  # noqa: E402


class TestNumberParser(unittest.TestCase):
    def test_commas_and_signs(self):
        self.assertEqual(g.parse_number("1,902"), 1902.0)
        self.assertEqual(g.parse_number("+449"), 449.0)
        self.assertEqual(g.parse_number("-545"), -545.0)
        self.assertEqual(g.parse_number("(-)"), None)

    def test_footnote_and_revision_suffix(self):
        self.assertEqual(g.parse_number("1,580 3/"), 1580.0)   # footnote marker
        self.assertEqual(g.parse_number("6,052r"), 6052.0)     # revised-figure suffix
        self.assertEqual(g.parse_number("$2,253"), 2253.0)

    def test_nulls(self):
        for s in ("", "-", "--", "n.a.", "..."):
            self.assertIsNone(g.parse_number(s))

    def test_parenthesized_negative(self):
        self.assertEqual(g.parse_number("(32)"), -32.0)


class TestMonthParser(unittest.TestCase):
    def test_abbreviations_and_year_prefix(self):
        self.assertEqual(g.month_of("1953-Jan."), (1, "1953"))
        self.assertEqual(g.month_of("Feb."), (2, None))
        self.assertEqual(g.month_of("Sept."), (9, None))
        self.assertEqual(g.month_of("December"), (12, None))

    def test_non_month(self):
        self.assertEqual(g.month_of("Cal. yr.")[0], None)
        self.assertEqual(g.month_of("1954 to date")[0], None)
        self.assertEqual(g.month_of("Total")[0], None)


class TestCategoryFilter(unittest.TestCase):
    def test_clean_strips_footnotes_quotes_dedup(self):
        self.assertEqual(g._clean_cat('Aid to agriculture 2h/'), "Aid to agriculture")
        self.assertEqual(g._clean_cat('Aid to agriculture 3/5/'), "Aid to agriculture")
        self.assertEqual(g._clean_cat('Aid to agriculture 2/ 3/'), "Aid to agriculture")  # space-sep
        self.assertEqual(g._clean_cat('"Other" expenditures'), "Other expenditures")
        self.assertEqual(g._clean_cat('debt_2'), "debt")

    def test_keeps_four_digit_year_in_name(self):
        self.assertEqual(g._clean_cat("Carriers' Taxing Act of 1937"),
                         "Carriers' Taxing Act of 1937")

    def test_good_categories(self):
        for c in ("National defense and related activities", "Army", "Aid to agriculture"):
            self.assertTrue(g.is_good_category(c), c)

    def test_rejects_junk(self):
        for c in ("col_15", "Change 1940 to 1941", "1937", "Total 11/",
                  "% of total interest-bearing debt", "1-5 years", "700 pounds each (head)",
                  "Adjustment for net difference due to reporting method", "Clearing account for checks"):
            self.assertFalse(g.is_good_category(c), c)


class TestFlowTableAndUnits(unittest.TestCase):
    def test_caption_and_units_and_flow(self):
        lines = [
            "Table 2.- Expenditures by Major Classifications",
            "(In millions of dollars)",
            "| Fiscal year or month | National defense |",
            "| --- | --- |",
            "| 1953-Jan. | 3,632 |",
        ]
        cap, units = g._caption_units(lines, 2)
        self.assertEqual(cap, "Expenditures by Major Classifications")
        self.assertEqual(units, "millions of dollars")
        t = g.Table(file="x.txt", header_line=2, headers=["m", "nd"], caption=cap, units=units)
        self.assertTrue(g.is_flow_table(t))

    def test_non_flow_caption_rejected(self):
        t = g.Table(file="x.txt", header_line=0, headers=[], caption="Public Debt by Maturity")
        self.assertFalse(g.is_flow_table(t))


class TestTemplatesTruthByConstruction(unittest.TestCase):
    def _canon(self, vec, year="1950", cat="Army", units="millions of dollars", calyr=None):
        cited = [{"line": 100 + i, "month": g.MONTHS[i], "raw": str(vec[i]), "value": float(vec[i])}
                 for i in range(12)]
        return g.Canon(category=cat, year=year, vector=[float(v) for v in vec], consistent=True,
                       n_sources=1, src_file="b.txt", src_header_line=99, cited_cells=cited,
                       calyr_total=calyr, units=units)

    def test_year_sum_and_calyr_xcheck(self):
        cc = self._canon(list(range(1, 13)), calyr=78.0)  # sum 1..12 = 78
        tr = g.t_year_sum(cc)
        self.assertEqual(tr["answer"], 78.0)
        self.assertEqual(tr["xcheck_abs_diff"], 0.0)
        self.assertIn("millions of dollars", tr["question"])

    def test_year_mean_halfyear_geomean(self):
        cc = self._canon([12] * 12)
        self.assertEqual(g.t_year_mean(cc)["answer"], 12.0)
        self.assertEqual(g.t_halfyear_mean(cc)["answer"], 12.0)          # Jul..Dec all 12
        self.assertEqual(g.t_year_geomean(cc)["answer"], 12.0)           # geomean of constants
        neg = self._canon([1, -1] + [1] * 10)
        self.assertIsNone(g.t_year_geomean(neg))                        # non-positive -> skipped

    def test_two_year_diff_and_pct(self):
        a = self._canon([10] * 12, year="1949")   # sum 120
        b = self._canon([12] * 12, year="1950")   # sum 144
        self.assertEqual(g.t_two_year_diff(a, b)["answer"], 24.0)
        self.assertEqual(g.t_two_year_pct(a, b)["answer"], 20.0)        # 24/120*100


class TestConsistencyGate(unittest.TestCase):
    def test_inconsistent_dropped_from_candidates(self):
        # same (cat, year), two different vectors across sources -> inconsistent -> not a candidate
        idx = {
            ("army", "1950"): g.Canon("Army", "1950", [1.0] * 12, consistent=False, n_sources=2,
                                      src_file="a", src_header_line=1, cited_cells=[], calyr_total=None),
            ("navy", "1950"): g.Canon("Navy", "1950", [2.0] * 12, consistent=True, n_sources=1,
                                      src_file="b", src_header_line=1,
                                      cited_cells=[{"line": 1, "month": m, "raw": "2", "value": 2.0}
                                                   for m in g.MONTHS], calyr_total=None),
        }
        cands = g.generate_candidates(idx)
        cats = {c["golden_path"]["category"] for c in cands}
        self.assertIn("Navy", cats)
        self.assertNotIn("Army", cats)


if __name__ == "__main__":
    unittest.main()
