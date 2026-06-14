"""The openrabbit findings JSON contract (SPEC section 8.1).

Every model and every output adapter speaks this DTO. The wire format uses
camelCase keys (``startLine``, ``endLine``, ``ruleId``) per the spec; the Python
:class:`Finding` dataclass uses snake_case attributes. ``to_dict``/``from_dict``
translate between the two.

This module has no network or cloud dependencies — ``jsonschema`` is the only
import and it is pure-Python.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

# Allowed enum vocabularies, kept here as the single source of truth and mirrored
# into the JSON schema below.
SEVERITIES = ("critical", "high", "medium", "low", "nit")
CATEGORIES = ("correctness", "security", "performance", "tests", "maintainability")
SIDES = ("LEFT", "RIGHT")

# Mapping between snake_case dataclass attributes and camelCase wire keys.
_FIELD_TO_WIRE = {
    "file": "file",
    "start_line": "startLine",
    "end_line": "endLine",
    "side": "side",
    "severity": "severity",
    "category": "category",
    "confidence": "confidence",
    "title": "title",
    "body": "body",
    "suggestion": "suggestion",
    "rule_id": "ruleId",
    "fingerprint": "fingerprint",
}
_WIRE_TO_FIELD = {v: k for k, v in _FIELD_TO_WIRE.items()}

_SCHEMA_PATH = Path(__file__).with_name("finding.schema.json")


@dataclass(frozen=True)
class Finding:
    """A single, structured review finding.

    Attributes mirror SPEC 8.1. ``side`` is ``LEFT`` or ``RIGHT`` (diff side),
    ``confidence`` is calibrated in ``[0, 1]``, and ``suggestion`` is an optional
    committable ```suggestion``` block.
    """

    file: str
    start_line: int
    end_line: int
    side: str  # one of SIDES
    severity: str  # one of SEVERITIES
    category: str  # one of CATEGORIES
    confidence: float  # 0..1
    title: str
    body: str
    rule_id: str
    fingerprint: str
    suggestion: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the camelCase wire contract (JSON-serializable)."""
        return {
            wire: getattr(self, attr) for attr, wire in _FIELD_TO_WIRE.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Finding":
        """Build a :class:`Finding` from a camelCase wire dict.

        Missing optional keys (``suggestion``) default to ``None``. Unknown keys
        are ignored so the contract can evolve additively.
        """
        kwargs: dict[str, Any] = {}
        for wire, attr in _WIRE_TO_FIELD.items():
            if wire in data:
                kwargs[attr] = data[wire]
        return cls(**kwargs)


def load_schema() -> dict[str, Any]:
    """Load and return the contract schema as a dict.

    The committed ``finding.schema.json`` is the loaded artifact; if it is
    absent (e.g. an unusual packaging), fall back to generating it from
    :func:`_schema_document` so the contract is always available. The
    ``test_schema_file_matches_generator`` test guards the two from drifting.
    """
    if _SCHEMA_PATH.exists():
        return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _schema_document()


def validate(finding_dict: dict[str, Any]) -> list[str]:
    """Validate a finding dict against the JSON schema.

    Returns a (possibly empty) list of human-readable error message strings.
    An empty list means the dict conforms to the contract. ``jsonschema`` is
    imported lazily to keep the contract module cheap to import.
    """
    import jsonschema  # local import; pure-python, no network

    schema = load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(finding_dict), key=lambda e: list(e.path))
    return [_format_error(e) for e in errors]


def _format_error(error: Any) -> str:
    location = "/".join(str(p) for p in error.path) or "<root>"
    return f"{location}: {error.message}"


def compute_fingerprint(file: str, rule_id: str, normalized_context: str) -> str:
    """Deterministic dedup/incremental fingerprint = sha256 hex.

    Per SPEC 8.1: ``sha256(file + ruleId + normalized-context)``. The context is
    whitespace-normalized (runs of whitespace collapsed to a single space, ends
    stripped) so cosmetic reformatting does not churn fingerprints. Fields are
    joined with a NUL separator so no concatenation collision is possible.
    """
    normalized = re.sub(r"\s+", " ", normalized_context).strip()
    payload = "\x00".join((file, rule_id, normalized))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ``_schema_document()`` is the single canonical generator of the contract: it
# derives the JSON Schema from the enum vocabularies above. The committed
# ``finding.schema.json`` is generated from it via :func:`write_schema_file`
# (an explicit build/CI step, NOT an import-time side effect) and a test asserts
# the two never drift.
def _schema_document() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://openrabbit.dev/finding.schema.json",
        "title": "openrabbit Finding",
        "description": "A single structured AI code-review finding (SPEC 8.1).",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "file",
            "startLine",
            "endLine",
            "side",
            "severity",
            "category",
            "confidence",
            "title",
            "body",
            "ruleId",
            "fingerprint",
        ],
        "properties": {
            "file": {"type": "string", "minLength": 1},
            "startLine": {"type": "integer", "minimum": 1},
            "endLine": {"type": "integer", "minimum": 1},
            "side": {"type": "string", "enum": list(SIDES)},
            "severity": {"type": "string", "enum": list(SEVERITIES)},
            "category": {"type": "string", "enum": list(CATEGORIES)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "title": {"type": "string", "minLength": 1},
            "body": {"type": "string"},
            "suggestion": {"type": ["string", "null"]},
            "ruleId": {"type": "string", "minLength": 1},
            "fingerprint": {"type": "string", "minLength": 1},
        },
    }


def write_schema_file() -> None:
    """(Re)generate the committed ``finding.schema.json`` from the canonical
    :func:`_schema_document`. Call this explicitly from a build/CI step after
    changing the enum vocabularies — it is intentionally NOT run on import, so
    a hand-edited JSON file can never silently disagree with the generator.
    """
    _SCHEMA_PATH.write_text(
        json.dumps(_schema_document(), indent=2) + "\n", encoding="utf-8"
    )
