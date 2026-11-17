from certiscan.extract import extract_document, merge_heats, parse_info, parse_table
from certiscan.textlayout import parse_number
from certiscan.types import Region, Word


def row(y, cells, h=20):
    """cells: list of (x, text). Each token becomes a word ~11px/char."""
    words = []
    for x, text in cells:
        cursor = x
        for tok in text.split():
            words.append(Word(tok, (cursor, y, cursor + 11 * len(tok), y + h)))
            cursor += 11 * len(tok) + 6
    return words


def test_info_inline_and_aligned_and_two_per_line():
    words = (
        row(0, [(0, "Certificate No.: MTC-2025-11111"), (900, "Date: 12-Mar-2025")])
        + row(40, [(0, "Heat No."), (300, "A12345, A12346")])
        + row(80, [(0, "Specification:"), (300, "ASTM A106/A106M-19a Gr. B")])
        + row(120, [(0, "Purchaser"), (300, "Apex Fabricators LLC"), (900, "Qty."), (1150, "12 PCS / 900 KG")])
    )
    info = parse_info(words)
    assert info["cert_no"] == "MTC-2025-11111"
    assert info["date"] == "12-Mar-2025"
    assert info["heat_list"] == "A12345, A12346"
    assert info["spec"] == "ASTM A106/A106M-19a Gr. B"
    assert info["customer"] == "Apex Fabricators LLC"
    assert info["quantity"] == "12 PCS / 900 KG"


def test_title_is_not_mistaken_for_certificate_number():
    words = row(0, [(0, "TEST CERTIFICATE EN 10204 3.1")]) + row(40, [(0, "Report No.:"), (300, "TC/1234/56")])
    assert parse_info(words)["cert_no"] == "TC/1234/56"


def test_chem_table_skips_limit_rows_and_maps_columns():
    xs = [0, 250, 400, 550, 700]
    words = (
        row(0, [(0, "CHEMICAL COMPOSITION (%)")])
        + row(50, list(zip(xs, ["Heat No.", "C", "Mn", "P", "S"])))
        + row(90, list(zip(xs, ["Max", "0.30", "1.06", "0.035", "0.035"])))
        + row(130, list(zip(xs, ["A12345", "0.18", "0.92", "0.012", "0.O08"])))  # OCR 'O' for zero
    )
    rows = parse_table(words, "chem")
    assert rows == [{"heat_no": "A12345", "values": {"C": 0.18, "Mn": 0.92, "P": 0.012, "S": 0.008}}]


def test_mech_table_converts_ksi_and_ignores_hardness():
    xs = [0, 250, 550, 850, 1100]
    words = (
        row(0, list(zip(xs, ["Heat #", "YS (ksi)", "UTS (ksi)", "El. (%)", "Hardness (HRB)"])))
        + row(40, list(zip(xs, ["A12345", "42.0", "70.5", "31", "82"])))
    )
    (r,) = parse_table(words, "mech")
    assert r["values"] == {"yield_mpa": round(42.0 * 6.894757, 1), "tensile_mpa": round(70.5 * 6.894757, 1),
                           "elongation_pct": 31.0}


def test_merged_ocr_phrase_is_split_across_columns():
    header = row(0, [(0, "Heat"), (200, "C"), (320, "Mn")])
    data = [Word("A12345", (0, 40, 66, 60)), Word("0.18 0.92", (195, 40, 360, 60))]
    (r,) = parse_table(header + data, "chem")
    assert r["values"] == {"C": 0.18, "Mn": 0.92}


def test_missing_one_letter_header_is_inferred_from_column_order():
    xs = [0, 250, 400, 550, 700]
    words = (
        row(0, list(zip([0, 400, 550, 700], ["Heat No.", "Mn", "P", "S"])))  # OCR dropped the "C" header
        + row(40, list(zip(xs, ["A12345", "0.18", "0.92", "0.012", "0.008"])))
    )
    (r,) = parse_table(words, "chem")
    assert r["values"] == {"C": 0.18, "Mn": 0.92, "P": 0.012, "S": 0.008}


def test_misread_ksi_header_is_caught_by_magnitude_check():
    xs = [0, 250, 550]
    words = row(0, list(zip(xs, ["Heat #", "YS (kpi)", "UTS (kpi)"]))) + row(40, list(zip(xs, ["A12345", "42.0", "70.5"])))
    (r,) = parse_table(words, "mech")
    assert r["values"]["yield_mpa"] == round(42.0 * 6.894757, 1)


def test_garbled_heat_numbers_snap_to_listed_heats():
    words = (
        row(0, [(0, "Heat No.:"), (300, "B01645")])
        + row(100, list(zip([0, 250, 400], ["Heat", "C", "Mn"])))
        + row(140, list(zip([0, 250, 400], ["BOL645", "0.19", "1.00"])))
    )
    regions = [Region("info_block", (-5, -5, 2000, 30)), Region("chem_table", (-5, 90, 2000, 170))]
    doc = extract_document(words, regions)
    assert [h["heat_no"] for h in doc["heats"]] == ["B01645"]


def test_merge_heats_tolerates_single_heat_ocr_mismatch():
    heats = merge_heats([{"heat_no": "A12345", "values": {"C": 0.2}}], [{"heat_no": "A12845", "values": {"yield_mpa": 300}}])
    assert heats == [{"heat_no": "A12345", "chemistry": {"C": 0.2}, "mechanical": {"yield_mpa": 300}}]


def test_extract_document_resolves_spec_key():
    words = row(0, [(0, "Spec.:"), (300, "ASME SA516")]) + row(40, [(0, "Grade:"), (300, "Gr. 70")])
    doc = extract_document(words, [Region("info_block", (-10, -10, 2000, 100))])
    assert doc["spec_key"] == "A516-70" and doc["grade"] == "70"


def test_parse_number_variants():
    assert parse_number(".035") == 0.035
    assert parse_number("0,18") == 0.18
    assert parse_number("62.5%") == 62.5
    assert parse_number("-") is None
