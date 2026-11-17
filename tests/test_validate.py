from certiscan.specs import resolve_spec_key
from certiscan.validate import validate_document


def doc(chem, mech, spec="ASTM A106 Gr. B"):
    return {"spec": spec, "grade": None, "heats": [{"heat_no": "H1", "chemistry": chem, "mechanical": mech}]}


GOOD_CHEM = {"C": 0.20, "Mn": 0.80, "P": 0.010, "S": 0.008, "Si": 0.22}
GOOD_MECH = {"yield_mpa": 300, "tensile_mpa": 480}


def test_pass():
    r = validate_document(doc(GOOD_CHEM, GOOD_MECH))
    assert r["status"] == "pass" and r["spec_key"] == "A106-B"


def test_fail_on_max_and_min():
    r = validate_document(doc({**GOOD_CHEM, "C": 0.33, "Mn": 0.20}, GOOD_MECH))
    assert r["status"] == "fail"
    assert {(v["field"], v["problem"]) for v in r["violations"]} == {("C", "above_max"), ("Mn", "below_min")}


def test_missing_required_needs_review():
    chem = {k: v for k, v in GOOD_CHEM.items() if k != "S"}
    r = validate_document(doc(chem, GOOD_MECH))
    assert r["status"] == "review" and r["missing"] == [{"heat_no": "H1", "field": "S"}]


def test_unknown_spec():
    assert validate_document(doc(GOOD_CHEM, GOOD_MECH, spec="EN 10216-2 P265GH"))["status"] == "unknown_spec"


def test_spec_key_resolution():
    assert resolve_spec_key("ASTM A106/A106M-19a", "Grade B") == "A106-B"
    assert resolve_spec_key("ASME SA516 GR 70", None) == "A516-70"
    assert resolve_spec_key("ASTM A36/A36M-19", None) == "A36"
