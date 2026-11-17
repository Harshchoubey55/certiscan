"""Rules-based compliance check of extracted values against spec limits."""
from __future__ import annotations

from typing import Optional

from certiscan.specs import load_specs, resolve_spec_key

MECH_FIELDS = ("yield_mpa", "tensile_mpa", "elongation_pct")


def _check_value(value, limit) -> Optional[str]:
    if "min" in limit and value < limit["min"]:
        return "below_min"
    if "max" in limit and value > limit["max"]:
        return "above_max"
    return None


def validate_document(doc: dict, specs: Optional[dict] = None) -> dict:
    """Return {'status', 'spec_key', 'violations', 'missing'}.

    status is one of:
      pass         - every required value present and every value within limits
      fail         - at least one value outside its limit
      review       - nothing out of limits, but a required value could not be read
      unknown_spec - the specification is not in the limits table
    """
    specs = specs if specs is not None else load_specs()
    key = doc.get("spec_key") or resolve_spec_key(doc.get("spec"), doc.get("grade"), specs)
    result = {"status": "unknown_spec", "spec_key": key, "violations": [], "missing": []}
    if key not in specs:
        return result
    spec = specs[key]
    heats = doc.get("heats") or []
    if not heats:
        result["status"] = "review"
        result["missing"].append({"heat_no": None, "field": "heats"})
        return result

    for heat in heats:
        heat_no = heat.get("heat_no")
        sections = (("chemistry", spec.get("chemistry", {})), ("mechanical", spec.get("mechanical", {})))
        for section, limits in sections:
            values = heat.get(section) or {}
            for name, limit in limits.items():
                value = values.get(name)
                if value is None:
                    if limit.get("required"):
                        result["missing"].append({"heat_no": heat_no, "field": name})
                    continue
                problem = _check_value(value, limit)
                if problem:
                    result["violations"].append(
                        {
                            "heat_no": heat_no,
                            "field": name,
                            "value": value,
                            "min": limit.get("min"),
                            "max": limit.get("max"),
                            "problem": problem,
                        }
                    )

    if result["violations"]:
        result["status"] = "fail"
    elif result["missing"]:
        result["status"] = "review"
    else:
        result["status"] = "pass"
    return result
