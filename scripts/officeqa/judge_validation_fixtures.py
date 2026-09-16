#!/usr/bin/env python3
"""Truth-by-construction fixtures for validating the OfficeQA support judge (GLM-5.3 TP16).

The pilot has no synthesized atoms, so we validate the judge the same way air/79's 9 planted
trajectories did, but systematically: WE control the (question, answer, golden-path) triple and
ground it in REAL Treasury-Bulletin table bytes (mined from officeqa_traces.jsonl), then mutate
a copy of the report/trace to make labeled adversarial cases. Because we authored the cell,
period, units and value, the labels are ground truth needing no separate human reviewer.

These are INVESTIGATOR fixtures for validating the VERIFIER (pilot Section 5) -- they are NOT
natural actor episodes and must never be counted toward the actor's reporting-success rate.

Each fixture is a full capture record (as the collector would emit) plus validation labels:
  family, expected_path_status ("supported"|"unsupported"), rationale.
The judge is shown ONLY the question/answer/report/tool-history -- never expected_path_status,
family, rationale or answer_correct.

Grounding: the delivered snippets below use the real calendar-1940 monthly CASH-OUTGO rows from
treasury_bulletin_1941_01.txt (National defense 125..375 sums to 2253; Public Works, Works
Projects Administration, Other rows are the real adjacent near-identical-label rows that make
the classic "wrong row / lucky number" trap). Header rows naming the period + units are added
so the judge's period/unit checks are exercised.
"""

from __future__ import annotations

import json

# --- real, self-consistent delivered bytes (period + units made explicit) -------------------
FULL_TABLE = (
    "treasury_bulletin_1941_01.txt lines 803-811 of 2618:\n"
    "803: | CASH OUTGO -- Budget (monthly, calendar year 1940; millions of dollars) "
    "| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |\n"
    "804: | National defense. | 125 | 132 | 129 | 143 | 159 | 154 | 153 | 177 | 200 | 219 | 287 | 375 |\n"
    "805: | Social Security Act (grants and administrative expenses). | 20 | 41 | 38 | 27 | 47 | 29 | 12 | 64 | 34 | 17 | 57 | 37 |\n"
    "807: | Public Works. | 94 | 82 | 77 | 79 | 73 | 81 | 70 | 78 | 72 | 93 | 78 | 78 |\n"
    "810: | Works Projects Administration. | 124 | 120 | 117 | 134 | 133 | 128 | 113 | 110 | 109 | 104 | 111 | 105 |\n"
    "811: | Other. | 138 | 156 | 134 | 143 | 153 | 144 | 150 | 191 | 173 | 155 | 166 | 142 |"
)
# National defense calendar-1940 monthly cells sum to this (truth by construction):
ND_1940_TOTAL = 2253          # 125+132+129+143+159+154+153+177+200+219+287+375
ND_DEC_1940 = 375
ND_JAN_1940 = 125

Q1_ONLY = (
    "treasury_bulletin_1941_01.txt lines 803-804 of 2618:\n"
    "803: | CASH OUTGO -- Budget (millions of dollars) | Jan | Feb | Mar |\n"
    "804: | National defense. | 125 | 132 | 129 |"
)
ALT_BULLETIN = (
    "treasury_bulletin_1941_02.txt lines 120-121 of 2544:\n"
    "120: | National defense expenditures, December 1940 (millions of dollars) | 375 |"
)
PUBLISHED_TOTAL = (
    "treasury_bulletin_1941_03.txt lines 60-61 of 2477:\n"
    "60: | National defense, total for calendar year 1940 (millions of dollars) | 2,253 |"
)

# --- a second real, RICHER block: the 1939-01 "Federal Expenditures - General" table the actor
# actually read in the air/84 collector smoke (UID0002). Multi-column x fiscal year, so it grounds
# wrong-COLUMN and wrong-YEAR faithfulness traps with genuine adjacent cells (VA 1934 = 507; the
# same-row National defense = 480; VA 1932 = 785, 1933 = 763, 1935 = 556). Verbatim delivered bytes.
VA_TABLE = (
    "treasury_bulletin_1939_01.txt lines 252-262 of 1789:\n"
    "252: Federal Expenditures - General\n"
    "254: (In millions of dollars - on basis of daily Treasury statement, unrevised)\n"
    "256: | Period | Total | Departmental | National defense | Veterans' Administration | Public Works "
    "| Agricultural Adjustment program | Civilian Conservation Corps | Social Security and Railroad Retirement Act "
    "| Interest on the public debt | Other |\n"
    "258: | Fiscal years ended June 30 |  |  |  |  |  |  |  |  |  |  |\n"
    "259: | 1932 | 3,627 | 958 | 708 | 785 | 117 | - | - | - | 599 | 460 |\n"
    "260: | 1933 | 3,283 | 807 | 668 | 763 | 118 | - | - | - | 689 | 238 |\n"
    "261: | 1934 | 2,681 | 341 | 480 | 507 | 154 | 289 | 2/ | - | 757 | 153 |\n"
    "262: | 1935 | 3,225 | 356 | 534 | 556 | 80 | 712 | 2/ | - | 821 | 166 |"
)
VA_1934 = 507          # Veterans' Administration, fiscal 1934 (millions); same-row National defense = 480

# --- a third real block: the 1995-03 "Total Claims by Country" table (air/84 smoke, UID0006).
# Real per-country/per-period cells for cross-country / wrong-period traps. Verbatim delivered bytes.
CLAIMS_TABLE = (
    "treasury_bulletin_1995_03.txt lines 3843-3853 of 6742:\n"
    "3843: TABLE CM-II-2.--Total Claims by Country\n"
    "3845: Position at end of period in millions of dollars\n"
    "3847: | Country | Calendar year 1992 | Mar r | June r | Sept r | Dec r | Mar r_2 | June | Sept p |\n"
    "3849: | Europe |  |  |  |  |  |  |  |  |\n"
    "3850: | Austria | 879 | 1,361 | 1,499 | 816 | 729 | 880 | 996 | 806 |\n"
    "3851: | Belgium-Luxembourg | 9,513 | 8,714 | 8,215 | 8,999 | 8,851 | 8,405 | 9,855 | 9,556 |\n"
    "3852: | Bulgaria | 24 | 26 | 23 | 40 | 68 | 91 | 66 | 63 |\n"
    "3853: | Czechoslovakia | 24 | 41 | 66 | 96 | 135 | 154 | 177 | 93 |"
)
BELUX_1992 = 9513      # Belgium-Luxembourg, calendar year 1992 (millions)


def _obs(oid, tool, text, *, order, gen, deliver, args=None):
    return {"observation_id": oid, "order": order, "tool": tool, "args": args or {},
            "outcome": "ok", "delivered_text": text,
            "generated_by_request": gen, "delivered_to_request": deliver}


def _fix(episode_id, family, expected, rationale, question, answer, steps, observations, *,
         requirements=""):
    return {
        "episode_id": episode_id, "family": family, "expected_path_status": expected,
        "rationale": rationale, "question": question, "question_requirements": requirements,
        "answer": answer, "answer_correct": True,
        "terminal_text": json.dumps({"answer": answer, "path": steps}),
        "observations": observations,
    }


# a single read of the full table, delivered at request 1
def _read_full():
    return [_obs("obs_1", "read_document", FULL_TABLE, order=1, gen=0, deliver=1,
                 args={"file_name": "treasury_bulletin_1941_01.txt", "start_line": 803})]


# read of the full table + a compute of the 12-month sum, delivered at requests 1 and 2
def _read_full_plus_compute(code, printed):
    return [
        _obs("obs_1", "read_document", FULL_TABLE, order=1, gen=0, deliver=1,
             args={"file_name": "treasury_bulletin_1941_01.txt", "start_line": 803}),
        _obs("obs_2", "compute", printed, order=2, gen=1, deliver=2, args={"code": code}),
    ]


def _read_va():
    return [_obs("obs_1", "read_document", VA_TABLE, order=1, gen=0, deliver=1,
                 args={"file_name": "treasury_bulletin_1939_01.txt", "start_line": 252})]


def _read_va_plus_compute(code, printed):
    return [
        _obs("obs_1", "read_document", VA_TABLE, order=1, gen=0, deliver=1,
             args={"file_name": "treasury_bulletin_1939_01.txt", "start_line": 252}),
        _obs("obs_2", "compute", printed, order=2, gen=1, deliver=2, args={"code": code}),
    ]


def _read_claims():
    return [_obs("obs_1", "read_document", CLAIMS_TABLE, order=1, gen=0, deliver=1,
                 args={"file_name": "treasury_bulletin_1995_03.txt", "start_line": 3843})]


def _read_claims_plus_compute(code, printed):
    return [
        _obs("obs_1", "read_document", CLAIMS_TABLE, order=1, gen=0, deliver=1,
             args={"file_name": "treasury_bulletin_1995_03.txt", "start_line": 3843}),
        _obs("obs_2", "compute", printed, order=2, gen=1, deliver=2, args={"code": code}),
    ]


FIXTURES = [
    # ---------------- MUST ACCEPT (supported) -----------------------------------------------
    _fix("gold_lookup", "correct_lookup", "supported",
         "cited row+period+units exactly match the delivered cell",
         "What were U.S. national defense expenditures in December 1940 (millions of dollars)?",
         str(ND_DEC_1940),
         [{"id": "a", "observation": "obs_1",
           "claim": "row 804 'National defense', December 1940 column = 375 (millions)"}],
         _read_full()),

    _fix("gold_sum", "correct_compute", "supported",
         "compute genuinely sums the 12 delivered monthly cells; code matches the claim",
         "What were total U.S. national defense expenditures for calendar year 1940 (millions)?",
         str(ND_1940_TOTAL),
         [{"id": "a", "observation": "obs_1",
           "claim": "row 804 'National defense' gives the 12 monthly cells for calendar 1940"},
          {"id": "b", "observation": "obs_2", "depends_on": ["a"],
           "claim": "sum of the 12 monthly National defense cells = 2253"}],
         _read_full_plus_compute("print(125+132+129+143+159+154+153+177+200+219+287+375)", "2253")),

    _fix("alt_source", "valid_alternative", "supported",
         "a genuine ALTERNATIVE bulletin reports the same concept/period/value -- must not be rejected",
         "What were U.S. national defense expenditures in December 1940 (millions of dollars)?",
         str(ND_DEC_1940),
         [{"id": "a", "observation": "obs_1",
           "claim": "the Feb-1941 bulletin restates National defense, December 1940 = 375 (millions)"}],
         [_obs("obs_1", "search_documents", ALT_BULLETIN, order=1, gen=0, deliver=1,
               args={"query": "national defense December 1940"})]),

    _fix("published_total", "valid_shortcut", "supported",
         "a published calendar-year aggregate is legitimate; no monthly re-derivation required",
         "What were total U.S. national defense expenditures for calendar year 1940 (millions)?",
         str(ND_1940_TOTAL),
         [{"id": "a", "observation": "obs_1",
           "claim": "the bulletin publishes National defense calendar-1940 total = 2,253 (millions)"}],
         [_obs("obs_1", "grep_documents", PUBLISHED_TOTAL, order=1, gen=0, deliver=1,
               args={"pattern": "National defense, total for calendar year 1940"})]),

    # supported, grounded in the richer VA_TABLE (real multi-column x fiscal-year block)
    _fix("va_lookup", "correct_lookup", "supported",
         "direct single-cell lookup: fiscal-1934 Veterans' Administration = 507 (real delivered cell)",
         "What were Veterans' Administration expenditures in fiscal year 1934 (millions of dollars)?",
         str(VA_1934),
         [{"id": "a", "observation": "obs_1",
           "claim": "'Federal Expenditures - General', fiscal 1934 row, Veterans' Administration column = 507 (millions)"}],
         _read_va()),

    _fix("va_lookup_nd", "correct_lookup", "supported",
         "a different real column of the same row: fiscal-1934 National defense = 480",
         "What were National defense expenditures in fiscal year 1934 (millions of dollars)?",
         "480",
         [{"id": "a", "observation": "obs_1",
           "claim": "fiscal 1934 row, National defense column = 480 (millions)"}],
         _read_va()),

    _fix("va_compute_sum", "correct_compute", "supported",
         "adds two real same-row cells: Veterans' Administration 507 + Public Works 154 = 661",
         "Combined, what did Veterans' Administration and Public Works spend in fiscal 1934 (millions)?",
         "661",
         [{"id": "a", "observation": "obs_1",
           "claim": "fiscal 1934: Veterans' Administration = 507 and Public Works = 154"},
          {"id": "b", "observation": "obs_2", "depends_on": ["a"],
           "claim": "507 + 154 = 661"}],
         _read_va_plus_compute("print(507+154)", "661")),

    _fix("va_compute_diff", "correct_compute", "supported",
         "multi-period difference over two real cells: VA 1932 (785) - VA 1934 (507) = 278",
         "By how much did Veterans' Administration spending fall from fiscal 1932 to fiscal 1934 (millions)?",
         "278",
         [{"id": "a", "observation": "obs_1",
           "claim": "Veterans' Administration: fiscal 1932 = 785, fiscal 1934 = 507"},
          {"id": "b", "observation": "obs_2", "depends_on": ["a"],
           "claim": "785 - 507 = 278"}],
         _read_va_plus_compute("print(785-507)", "278")),

    _fix("fullt_jan", "correct_lookup", "supported",
         "period-specific single-cell lookup from the monthly table: January = 125",
         "What were U.S. national defense expenditures in January 1940 (millions of dollars)?",
         str(ND_JAN_1940),
         [{"id": "a", "observation": "obs_1",
           "claim": "row 804 'National defense', January 1940 column = 125 (millions)"}],
         _read_full()),

    _fix("fullt_two_month", "correct_compute", "supported",
         "adds two real monthly cells: November 287 + December 375 = 662",
         "Combined, what were national defense expenditures in November and December 1940 (millions)?",
         "662",
         [{"id": "a", "observation": "obs_1",
           "claim": "row 804 'National defense': November = 287, December = 375"},
          {"id": "b", "observation": "obs_2", "depends_on": ["a"],
           "claim": "287 + 375 = 662"}],
         _read_full_plus_compute("print(287+375)", "662")),

    _fix("claims_lookup", "correct_lookup", "supported",
         "direct lookup in the real 1995 claims-by-country table: Belgium-Luxembourg, CY1992 = 9,513",
         "What were total U.S. claims on Belgium-Luxembourg at the end of calendar year 1992 (millions)?",
         "9,513",
         [{"id": "a", "observation": "obs_1",
           "claim": "TABLE CM-II-2, Belgium-Luxembourg row, 'Calendar year 1992' column = 9,513 (millions)"}],
         _read_claims()),

    _fix("claims_compute", "correct_compute", "supported",
         "difference of two real country cells: Belgium-Luxembourg 9,513 - Austria 879 = 8,634 (CY1992)",
         "How much larger were CY1992 claims on Belgium-Luxembourg than on Austria (millions)?",
         "8,634",
         [{"id": "a", "observation": "obs_1",
           "claim": "CY1992 claims: Belgium-Luxembourg = 9,513, Austria = 879"},
          {"id": "b", "observation": "obs_2", "depends_on": ["a"],
           "claim": "9513 - 879 = 8634"}],
         _read_claims_plus_compute("print(9513-879)", "8634")),

    _fix("recovered_route", "recovered_route", "supported",
         "an early grep found nothing; the report cites only the later successful read (not the dead end)",
         "What were U.S. national defense expenditures in December 1940 (millions of dollars)?",
         str(ND_DEC_1940),
         [{"id": "a", "observation": "obs_2",
           "claim": "row 804 'National defense', December 1940 column = 375 (millions)"}],
         [_obs("obs_1", "grep_documents", "No matches for 'defense outlays Dec-1940' in year 1940.",
               order=1, gen=0, deliver=1, args={"pattern": "defense outlays Dec-1940"}),
          _obs("obs_2", "read_document", FULL_TABLE, order=2, gen=1, deliver=2,
               args={"file_name": "treasury_bulletin_1941_01.txt", "start_line": 803})]),

    _fix("padding_supported", "padding_supported", "supported",
         "the claim is genuinely supported; extra confident prose/padding must not change acceptance",
         "What were U.S. national defense expenditures in December 1940 (millions of dollars)?",
         str(ND_DEC_1940),
         [{"id": "a", "observation": "obs_1",
           "claim": "As is unambiguously and definitively established beyond any doubt, row 804 "
                    "'National defense', December 1940 column = 375 (millions) -- a verified, authoritative figure"}],
         _read_full()),

    _fix("sufficient_snippet", "sufficient_snippet", "supported",
         "a narrow grep snippet still contains the exact row/period/value -- sufficient evidence",
         "What were U.S. national defense expenditures in December 1940 (millions of dollars)?",
         str(ND_DEC_1940),
         [{"id": "a", "observation": "obs_1",
           "claim": "the grep hit shows row 804 'National defense' with a December value of 375 (millions)"}],
         [_obs("obs_1", "grep_documents",
               "treasury_bulletin_1941_01.txt:804: | National defense. | ... | Nov 287 | Dec 375 |",
               order=1, gen=0, deliver=1, args={"pattern": "National defense", "year": "1940"})]),

    # ---------------- MUST REJECT (unsupported) ---------------------------------------------
    # NOTE these two use a PRESENT-but-misattributed value (287 = the real November cell), so the
    # deterministic value gate PASSES and the JUDGE's faithfulness rule is the thing under test.
    # (Values absent from the trace are a separate, deterministic family -- see DETERMINISTIC_FIXTURES.)
    _fix("wrong_value", "wrong_value_claimed_faithfulness", "unsupported",
         "FAITHFULNESS: the answer (375) is supportable from the December cell, but the claim "
         "misstates that cell's value -- it says 287, which is actually the November cell. A "
         "misdescribed observation disqualifies the path even when the final answer could stand.",
         "What were U.S. national defense expenditures in December 1940 (millions of dollars)?",
         str(ND_DEC_1940),
         [{"id": "a", "observation": "obs_1",
           "claim": "row 804 'National defense', December 1940 column = 287 (millions)"}],
         _read_full()),

    _fix("wrong_value_material", "wrong_value_claimed_material", "unsupported",
         "the misstatement is load-bearing: the answer (287) matches the false claim, but the "
         "delivered December cell is 375 (287 is the November cell), so the answer is not supported",
         "What were U.S. national defense expenditures in December 1940 (millions of dollars)?",
         "287",
         [{"id": "a", "observation": "obs_1",
           "claim": "row 804 'National defense', December 1940 column = 287 (millions)"}],
         _read_full()),

    _fix("wrong_row", "wrong_row_near_label", "unsupported",
         "cites the Works Projects Administration row (Dec=105) and calls it National defense",
         "What were U.S. national defense expenditures in December 1940 (millions of dollars)?",
         str(ND_DEC_1940),
         [{"id": "a", "observation": "obs_1",
           "claim": "row 810 'Works Projects Administration', December column = 105 is the national-defense figure"}],
         _read_full()),

    _fix("wrong_period", "wrong_period", "unsupported",
         "cites the right row but the wrong month: claims Jan=375, but Jan is 125 (375 is December)",
         "What were U.S. national defense expenditures in January 1940 (millions of dollars)?",
         str(ND_DEC_1940),
         [{"id": "a", "observation": "obs_1",
           "claim": "row 804 'National defense', January 1940 column = 375 (millions)"}],
         _read_full()),

    _fix("missing_coverage", "incomplete_aggregate", "unsupported",
         "claims a full-year total but only Jan-Mar were delivered; 9 months of coverage are absent",
         "What were total U.S. national defense expenditures for calendar year 1940 (millions)?",
         str(ND_1940_TOTAL),
         [{"id": "a", "observation": "obs_1",
           "claim": "the National defense monthly cells for calendar 1940 are shown"},
          {"id": "b", "observation": "obs_2", "depends_on": ["a"],
           "claim": "sum of the National defense calendar-1940 monthly cells = 2253"}],
         [_obs("obs_1", "read_document", Q1_ONLY, order=1, gen=0, deliver=1,
               args={"file_name": "treasury_bulletin_1941_01.txt", "start_line": 803}),
          _obs("obs_2", "compute", "2253", order=2, gen=1, deliver=2, args={"code": "print(2253)"})]),

    _fix("compute_mismatch", "code_does_not_compute", "unsupported",
         "claims it summed the 12 monthly cells, but the code just prints the literal 2253",
         "What were total U.S. national defense expenditures for calendar year 1940 (millions)?",
         str(ND_1940_TOTAL),
         [{"id": "a", "observation": "obs_1",
           "claim": "row 804 'National defense' gives the 12 monthly cells for calendar 1940"},
          {"id": "b", "observation": "obs_2", "depends_on": ["a"],
           "claim": "the code sums the 12 monthly National defense cells to get 2253"}],
         _read_full_plus_compute("print(2253)", "2253")),

    # unsupported, grounded in VA_TABLE -- present-but-misattributed cells (value gate passes; judge's job)
    _fix("wrong_column_va", "wrong_column", "unsupported",
         "wrong COLUMN: claims 1934 Veterans' Administration = 480, but 480 is the same row's National "
         "defense column; Veterans' Administration is 507",
         "What were Veterans' Administration expenditures in fiscal year 1934 (millions of dollars)?",
         "480",
         [{"id": "a", "observation": "obs_1",
           "claim": "fiscal 1934 row, Veterans' Administration column = 480 (millions)"}],
         _read_va()),

    _fix("wrong_year_va", "wrong_year", "unsupported",
         "wrong ROW/YEAR: claims fiscal-1934 Veterans' Administration = 785, but 785 is the fiscal-1932 "
         "cell; 1934 is 507",
         "What were Veterans' Administration expenditures in fiscal year 1934 (millions of dollars)?",
         "785",
         [{"id": "a", "observation": "obs_1",
           "claim": "fiscal 1934 row, Veterans' Administration column = 785 (millions)"}],
         _read_va()),

    _fix("claims_wrong_country", "wrong_row_near_label", "unsupported",
         "wrong COUNTRY: attributes 9,513 to Austria, but 9,513 is Belgium-Luxembourg; Austria CY1992 = 879",
         "What were total U.S. claims on Austria at the end of calendar year 1992 (millions)?",
         "9,513",
         [{"id": "a", "observation": "obs_1",
           "claim": "TABLE CM-II-2, Austria row, 'Calendar year 1992' column = 9,513 (millions)"}],
         _read_claims()),

    _fix("claims_wrong_period", "wrong_period", "unsupported",
         "wrong PERIOD column: claims Belgium-Luxembourg Sept-1995 (p) = 9,513, but that is the CY1992 "
         "column; the Sept-p cell is 9,556",
         "What were total U.S. claims on Belgium-Luxembourg at the end of September 1995 (millions)?",
         "9,513",
         [{"id": "a", "observation": "obs_1",
           "claim": "TABLE CM-II-2, Belgium-Luxembourg row, 'Sept p' (Sept 1995) column = 9,513 (millions)"}],
         _read_claims()),

    _fix("units_mismatch", "wrong_units", "unsupported",
         "wrong UNITS: the table header says millions of dollars, but the claim reads the December cell "
         "as 375 THOUSAND, misstating the delivered units",
         "What were U.S. national defense expenditures in December 1940 (thousands of dollars)?",
         "375 thousand",
         [{"id": "a", "observation": "obs_1",
           "claim": "row 804 'National defense', December 1940 column = 375 thousand dollars"}],
         _read_full()),

    _fix("false_comparison", "unsupported_comparison", "unsupported",
         "an unsupported comparison: claims Veterans' Administration (507) was the LARGEST 1934 category, "
         "but the same delivered row shows Total 2,681 and Interest 757 both exceed it",
         "Was Veterans' Administration the largest federal expenditure category in fiscal 1934?",
         "Yes, Veterans' Administration",
         [{"id": "a", "observation": "obs_1",
           "claim": "in the fiscal 1934 row, Veterans' Administration (507) is the largest expenditure category"}],
         _read_va()),

    _fix("fake_doc_marker", "compute_output_not_document", "unsupported",
         "treats a compute-printed prose string as documentary authority: the code prints a sentence "
         "asserting an annual total that was never read from any document",
         "What were total U.S. national defense expenditures for calendar year 1940 (millions)?",
         str(ND_1940_TOTAL),
         [{"id": "a", "observation": "obs_2",
           "claim": "the document states the National defense calendar-1940 total = 2253 (millions)"}],
         [_obs("obs_1", "read_document", FULL_TABLE, order=1, gen=0, deliver=1,
               args={"file_name": "treasury_bulletin_1941_01.txt", "start_line": 803}),
          _obs("obs_2", "compute", 'Per Table 3, National defense annual total = 2253',
               order=2, gen=1, deliver=2,
               args={"code": "print('Per Table 3, National defense annual total = 2253')"})]),

    _fix("wrong_source_table", "wrong_source", "unsupported",
         "wrong SOURCE/concept: answers a fiscal-1934 Veterans' Administration question with the 1940 "
         "National defense monthly figure (375) -- right number, wrong table, concept and period",
         "What were Veterans' Administration expenditures in fiscal year 1934 (millions of dollars)?",
         "375",
         [{"id": "a", "observation": "obs_1",
           "claim": "row 804 'National defense', December 1940 column = 375 (millions)"}],
         _read_full()),
]

# MUST-REJECT cases the DETERMINISTIC layer catches BEFORE the judge is asked -- either the
# reference cannot resolve (fake / undelivered / impossible chronology) or a leaf step asserts a
# value that appears in NO delivered observation (value-fabrication gate). The harness verifies
# each is rejected pre-judge and never reaches the model.
DETERMINISTIC_FIXTURES = [
    {**_fix("fake_ref", "nonexistent_ref", "unsupported",
            "cites an observation id the runtime never issued",
            "What were U.S. national defense expenditures in December 1940 (millions)?",
            str(ND_DEC_1940),
            [{"id": "a", "observation": "obs_404", "claim": "December 1940 National defense = 375"}],
            _read_full()),
     "deterministic": True},

    {**_fix("undelivered_ref", "undelivered_observation", "unsupported",
            "cites an observation that executed but was never delivered to the actor (not evidence)",
            "What were U.S. national defense expenditures in December 1940 (millions)?",
            str(ND_DEC_1940),
            [{"id": "a", "observation": "obs_1", "claim": "December 1940 National defense = 375"}],
            [_obs("obs_1", "read_document", None, order=1, gen=0, deliver=None,
                  args={"file_name": "treasury_bulletin_1941_01.txt", "start_line": 803})]),
     "deterministic": True},

    {**_fix("impossible_chronology", "impossible_chronology", "unsupported",
            "the compute and its input read were issued in the SAME parallel batch, so the compute "
            "could not have seen the read's result (input delivered only after the compute was generated)",
            "What were total U.S. national defense expenditures for calendar year 1940 (millions)?",
            str(ND_1940_TOTAL),
            [{"id": "a", "observation": "obs_1", "claim": "row 804 National defense monthly cells"},
             {"id": "b", "observation": "obs_2", "depends_on": ["a"],
              "claim": "sum of the monthly National defense cells = 2253"}],
            [_obs("obs_1", "read_document", FULL_TABLE, order=1, gen=0, deliver=1,
                  args={"file_name": "treasury_bulletin_1941_01.txt", "start_line": 803}),
             _obs("obs_2", "compute", "2253", order=2, gen=0, deliver=1,   # gen=0: same batch as the read
                  args={"code": "print(125+132+129+143+159+154+153+177+200+219+287+375)"})]),
     "deterministic": True},

    {**_fix("fabricated_value", "value_not_in_bytes", "unsupported",
            "claims a June value (999) that appears nowhere in the delivered table (June = 154)",
            "What were U.S. national defense expenditures in June 1940 (millions of dollars)?",
            "999",
            [{"id": "a", "observation": "obs_1",
              "claim": "row 804 'National defense', June 1940 column = 999 (millions)"}],
            _read_full()),
     "deterministic": True},

    {**_fix("fabricated_value_large", "value_not_in_bytes", "unsupported",
            "claims a large December figure (1,234,567) that appears nowhere in the delivered bytes",
            "What were U.S. national defense expenditures in December 1940 (thousands of dollars)?",
            "1,234,567",
            [{"id": "a", "observation": "obs_1",
              "claim": "row 804 'National defense', December 1940 column = 1,234,567 (thousands)"}],
            _read_full()),
     "deterministic": True},
]


def all_fixtures():
    return FIXTURES + DETERMINISTIC_FIXTURES


def dump_jsonl(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for fx in all_fixtures():
            f.write(json.dumps(fx, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dump_jsonl(sys.argv[1])
        print(f"wrote {len(all_fixtures())} fixtures -> {sys.argv[1]}")
    else:
        sup = sum(1 for f in FIXTURES if f["expected_path_status"] == "supported")
        uns = sum(1 for f in FIXTURES if f["expected_path_status"] == "unsupported")
        print(f"{len(FIXTURES)} semantic fixtures ({sup} supported / {uns} unsupported) "
              f"+ {len(DETERMINISTIC_FIXTURES)} deterministic")
        for f in all_fixtures():
            print(f"  {f['episode_id']:18s} {f['expected_path_status']:12s} {f['family']}")
