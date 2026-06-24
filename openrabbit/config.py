"""Loader for ``.openrabbit.yaml`` config-as-code (SPEC section 8.2).

``load_config`` accepts either a filesystem path (``str``/``Path``) or an
already-parsed mapping and returns a typed, defaulted, validated :class:`Config`.
Unknown keys are tolerated additively where the spec says the format is
"additive/versioned", but enum-constrained values are validated strictly so a
typo fails fast rather than silently degrading review quality.

``pyyaml`` is imported lazily inside :func:`load_config` so the module imports
with no third-party deps when callers pass a dict.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union, overload

from openrabbit.bedrock_models import Severity, Verdict, validate_model_region

# --------------------------------------------------------------------------- #
# vocabularies & defaults                                                     #
# --------------------------------------------------------------------------- #
PROFILES = ("chill", "balanced", "assertive")
LENSES = ("correctness", "security", "performance", "tests", "maintainability")
TELEMETRY_MODES = ("opt-in", "opt-out")
# Per-lens finder reasoning effort (Nova 2 extended-thinking). Maps onto the
# Converse ``additionalModelRequestFields.reasoningConfig.maxReasoningEffort``.
# A lens absent from ``review.lens_reasoning_effort`` runs with reasoning OFF
# (the Nova 2 default ``type: "disabled"``).
REASONING_EFFORTS = ("low", "medium", "high")
# Canonical user-facing response languages. The harness normalizes friendly
# aliases (any case) down to one of these 2-letter codes, so the prompt/label
# logic only ever sees a canonical value. ``en`` is the default (no behavior
# change). Extend by adding a canonical code here + an alias mapping below + a
# Korean-style label map in walkthrough.py.
RESPONSE_LANGUAGES = ("en", "ko")
DEFAULT_RESPONSE_LANGUAGE = "en"
# Rabbit persona / branding (Feature 2). When ON (the default) the posted
# walkthrough carries tasteful 🐰 branding + a one-line sign-off; when OFF the
# output is plain/neutral so teams can opt out. A bool keeps the knob trivial.
DEFAULT_PERSONA = True
# Friendly aliases -> canonical code. Keys are matched case-insensitively.
_RESPONSE_LANGUAGE_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "ko": "ko",
    "kor": "ko",
    "korea": "ko",
    "korean": "ko",
    "한국어": "ko",
}
# Severity vocabulary ordered most→least severe; index = rank (0 = critical).
# Mirrors openrabbit.findings.SEVERITIES (kept independent here so config has no
# import-time dependency on the findings contract).
SEVERITIES = ("critical", "high", "medium", "low", "nit")

DEFAULT_CONFIDENCE_GATE = 0.80
DEFAULT_PROFILE = "balanced"
DEFAULT_LENSES = list(LENSES)
# Only HIGH/CRITICAL findings route through the expensive cross-family verifier
# by default; everything below takes the cheaper finder-confidence path (SPEC
# 7.3 cost lever #3). Widen via review.verify_min_severity.
DEFAULT_VERIFY_MIN_SEVERITY = "high"
# TRUST-CORE lenses: findings in these categories route through the verifier
# REGARDLESS of severity (verify-strict thesis). A hallucinated medium/correctness
# finding is thus actually re-checked (and refuted) instead of posting blind via
# the cheap finder-confidence path. Override via review.always_verify_categories.
DEFAULT_ALWAYS_VERIFY_CATEGORIES = frozenset({"correctness", "security"})
# Findings that STILL bypass the verifier (below verify_min_severity AND not
# trust-core) must clear a HIGHER finder-confidence bar than the normal gate to
# post UN-verified — so low/maintainability nitpicks at ~0.8 are dropped rather
# than posted unverified (low-noise). Must be >= confidence_gate.
DEFAULT_UNVERIFIED_CONFIDENCE_GATE = 0.9

PathLike = Union[str, Path]


class ConfigError(ValueError):
    """Raised when a config is missing, unparsable, or invalid."""


# --------------------------------------------------------------------------- #
# typed config model                                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PathInstruction:
    """A per-path reviewer instruction (SPEC 8.2 ``path_instructions``)."""

    path: str
    instructions: str


@dataclass(frozen=True)
class ReviewConfig:
    """The ``review:`` block."""

    profile: str = DEFAULT_PROFILE
    confidence_gate: float = DEFAULT_CONFIDENCE_GATE
    incremental: bool = True
    path_filters: list[str] = field(default_factory=list)
    path_instructions: list[PathInstruction] = field(default_factory=list)
    lenses: list[str] = field(default_factory=lambda: list(DEFAULT_LENSES))
    #: Minimum severity routed through the (expensive) cross-family verifier;
    #: less-severe findings take the cheaper finder-confidence path.
    verify_min_severity: str = DEFAULT_VERIFY_MIN_SEVERITY
    #: TRUST-CORE categories routed through the verifier REGARDLESS of severity
    #: (verify-strict). A finding is verified if it is at least as severe as
    #: ``verify_min_severity`` OR its category is in this set.
    always_verify_categories: frozenset[str] = field(
        default_factory=lambda: frozenset(DEFAULT_ALWAYS_VERIFY_CATEGORIES)
    )
    #: Higher finder-confidence bar a finding must clear to post UN-verified (when
    #: it bypassed the verifier). Must be >= ``confidence_gate``; dropping noisy
    #: low-severity nitpicks that the verifier never vetted.
    unverified_confidence_gate: float = DEFAULT_UNVERIFIED_CONFIDENCE_GATE
    #: Per-lens finder reasoning effort (``{lens: "low"|"medium"|"high"}``). A
    #: lens omitted here runs with reasoning OFF. ``run_lenses`` threads each
    #: configured effort into the finder ``complete()`` call as
    #: ``opts['reasoning_effort']``, which the Converse adapter maps onto
    #: ``reasoningConfig.maxReasoningEffort`` (research: LOW for
    #: correctness/security/logic/concurrency lenses, OFF for style/tests).
    lens_reasoning_effort: dict[str, str] = field(default_factory=dict)
    #: Canonical user-facing response language (``"en"`` default or ``"ko"``).
    #: When non-``en`` the finder + verifier prompts gain a language instruction
    #: ("write each finding's user-facing title/description in <lang>; REASON in
    #: English") and the walkthrough's static labels render in that language.
    #: Aliases (e.g. ``"korean"``, ``"KO"``) are normalized to the canonical code
    #: at load time, so this field is always one of :data:`RESPONSE_LANGUAGES`.
    response_language: str = DEFAULT_RESPONSE_LANGUAGE
    #: Rabbit persona / branding toggle (Feature 2). ``True`` (default) brands the
    #: walkthrough with a 🐰 in the heading + a one-line sign-off (localized to
    #: :attr:`response_language`); ``False`` produces plain, neutral output so a
    #: team can opt out. Threaded from the orchestrator into the walkthrough
    #: renderer exactly like :attr:`response_language`.
    persona: bool = DEFAULT_PERSONA


@dataclass(frozen=True)
class ModelRole:
    """A single ``model_roles.<role>`` entry: BYO Bedrock model + region + opts.

    ``model`` and ``region`` are first-class; any other keys (``reasoning_effort``,
    ``store``, ``enabled``, ...) are preserved in :attr:`options`.
    """

    model: str
    region: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalTools:
    """The ``external_tools:`` block — deterministic graders fed into context."""

    enabled: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Telemetry:
    """The ``telemetry:`` block."""

    enabled: bool = True
    mode: str = "opt-out"


@dataclass(frozen=True)
class Config:
    """Parsed, validated ``.openrabbit.yaml`` document."""

    version: int = 1
    review: ReviewConfig = field(default_factory=ReviewConfig)
    model_roles: dict[str, ModelRole] = field(default_factory=dict)
    external_tools: ExternalTools = field(default_factory=ExternalTools)
    telemetry: Telemetry = field(default_factory=Telemetry)


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #
@overload
def load_config(source: Union[PathLike, Mapping[str, Any]]) -> Config: ...
@overload
def load_config(
    source: Union[PathLike, Mapping[str, Any]], *, collect_warnings: bool
) -> Union[Config, tuple[Config, list[str]]]: ...


def load_config(
    source: Union[PathLike, Mapping[str, Any]], *, collect_warnings: bool = False
) -> Union[Config, tuple[Config, list[str]]]:
    """Load and validate a config from a path or an already-parsed mapping.

    Raises :class:`ConfigError` for missing files, parse errors, invalid values,
    or **hard** ``model_roles`` problems (e.g. GPT-5.5 in an unsupported region —
    its endpoint does not exist there). Soft ``model_roles`` issues (unknown
    model id, off-allow-list region for a Converse model) never block the load.

    By default returns the :class:`Config`. Pass ``collect_warnings=True`` to get
    ``(Config, warnings)`` so callers can surface soft warnings (CLI/log).
    """
    if isinstance(source, Mapping):
        data: Any = dict(source)
    elif isinstance(source, (str, Path)):
        data = _read_yaml(source)
    else:
        raise ConfigError(
            "config source must be a path (str/Path) or a mapping, got "
            f"{type(source).__name__}"
        )

    if not isinstance(data, Mapping):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")

    config = _parse(dict(data))

    # Validate model_roles against Bedrock allow-lists/regions. Hard errors raise
    # immediately; soft warnings are returned only when the caller asks.
    errors, warnings = _classify_model_role_verdicts(config)
    if errors:
        raise ConfigError("; ".join(errors))

    if collect_warnings:
        return config, warnings
    return config


def _iter_model_role_verdicts(config: Config) -> Iterator[tuple[str, Verdict]]:
    """Yield ``(role, verdict)`` for each role with a non-``None`` verdict.

    The single source of truth for the per-role model/region check: every
    ``model_roles`` consumer (:func:`validate_model_roles`,
    :func:`_classify_model_role_verdicts`) iterates this so the check (and its
    sort order) lives in exactly one place. Roles are visited in sorted order.
    """
    for role, spec in sorted(config.model_roles.items()):
        verdict = validate_model_region(spec.model, spec.region)
        if verdict is not None:
            yield role, verdict


def validate_model_roles(config: Config) -> list[str]:
    """Return human-readable problems with ``config.model_roles`` (SPEC 7.2/8.2).

    Each role's model id is checked against :mod:`openrabbit.bedrock_models`:
    the model must be recognized (else a warning), its region must be in the
    model's allow-list (a **hard error** for GPT-5.5 outside us-east-1/2; a
    warning for soft-region Converse models), and so on. Messages are prefixed
    with their severity and role so a caller (CLI/log) can present them directly.

    This is the soft, *non-raising* surface — it returns *all* problems (warnings
    and errors). :func:`load_config` consults the underlying verdicts to decide
    what raises; a clean config yields ``[]``.
    """
    return [
        f"[{verdict.severity.value}] model_roles.{role}: {verdict.message}"
        for role, verdict in _iter_model_role_verdicts(config)
    ]


def _classify_model_role_verdicts(config: Config) -> tuple[list[str], list[str]]:
    """Split model-role verdicts into ``(hard_errors, soft_warnings)``.

    Same checks as :func:`validate_model_roles` (shared via
    :func:`_iter_model_role_verdicts`) but keyed on the structured
    :class:`~openrabbit.bedrock_models.Severity` so :func:`load_config` can raise
    on errors while merely collecting warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    for role, verdict in _iter_model_role_verdicts(config):
        msg = f"model_roles.{role}: {verdict.message}"
        if verdict.severity is Severity.ERROR:
            errors.append(msg)
        else:
            warnings.append(f"[{verdict.severity.value}] {msg}")
    return errors, warnings


# --------------------------------------------------------------------------- #
# internals                                                                   #
# --------------------------------------------------------------------------- #
def _read_yaml(path: PathLike) -> Any:
    import yaml  # local import: pure-python parser, no network

    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"failed to parse YAML at {p}: {exc}") from exc


def _parse(data: dict[str, Any]) -> Config:
    version = data.get("version", 1)
    if not isinstance(version, int):
        raise ConfigError("version must be an integer")

    return Config(
        version=version,
        review=_parse_review(data.get("review", {}) or {}),
        model_roles=_parse_model_roles(data.get("model_roles", {}) or {}),
        external_tools=_parse_external_tools(data.get("external_tools", {}) or {}),
        telemetry=_parse_telemetry(data.get("telemetry", {}) or {}),
    )


def _require_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{what} must be a mapping")
    return dict(value)


def _parse_review(raw: Any) -> ReviewConfig:
    block = _require_mapping(raw, "review")

    profile = block.get("profile", DEFAULT_PROFILE)
    if profile not in PROFILES:
        raise ConfigError(f"review.profile must be one of {PROFILES}, got {profile!r}")

    gate = block.get("confidence_gate", DEFAULT_CONFIDENCE_GATE)
    if not isinstance(gate, (int, float)) or isinstance(gate, bool):
        raise ConfigError("review.confidence_gate must be a number")
    if not 0.0 <= float(gate) <= 1.0:
        raise ConfigError("review.confidence_gate must be within [0, 1]")

    incremental = block.get("incremental", True)
    if not isinstance(incremental, bool):
        raise ConfigError("review.incremental must be a boolean")

    path_filters = block.get("path_filters", []) or []
    if not _is_str_list(path_filters):
        raise ConfigError("review.path_filters must be a list of strings")

    lenses = block.get("lenses")
    if lenses is None:
        lenses = list(DEFAULT_LENSES)
    if not _is_str_list(lenses):
        raise ConfigError("review.lenses must be a list of strings")
    for lens in lenses:
        if lens not in LENSES:
            raise ConfigError(
                f"review.lenses contains unknown lens {lens!r}; allowed: {LENSES}"
            )

    instructions = _parse_path_instructions(block.get("path_instructions", []) or [])

    verify_min_severity = block.get("verify_min_severity", DEFAULT_VERIFY_MIN_SEVERITY)
    if verify_min_severity not in SEVERITIES:
        raise ConfigError(
            "review.verify_min_severity must be one of "
            f"{SEVERITIES}, got {verify_min_severity!r}"
        )

    always_verify_categories = _parse_always_verify_categories(
        block.get("always_verify_categories")
    )

    unverified_gate = _parse_unverified_confidence_gate(block, float(gate))

    lens_reasoning_effort = _parse_lens_reasoning_effort(
        block.get("lens_reasoning_effort", {}) or {}
    )

    response_language = _parse_response_language(
        block.get("response_language", DEFAULT_RESPONSE_LANGUAGE)
    )

    persona = block.get("persona", DEFAULT_PERSONA)
    if not isinstance(persona, bool):
        raise ConfigError("review.persona must be a boolean")

    return ReviewConfig(
        profile=profile,
        confidence_gate=float(gate),
        incremental=incremental,
        path_filters=list(path_filters),
        path_instructions=instructions,
        lenses=list(lenses),
        verify_min_severity=verify_min_severity,
        always_verify_categories=always_verify_categories,
        unverified_confidence_gate=unverified_gate,
        lens_reasoning_effort=lens_reasoning_effort,
        response_language=response_language,
        persona=persona,
    )


def _parse_response_language(raw: Any) -> str:
    """Parse + normalize ``review.response_language`` to a canonical code.

    Accepts a string alias (case-insensitive) from
    :data:`_RESPONSE_LANGUAGE_ALIASES` and returns the canonical 2-letter code in
    :data:`RESPONSE_LANGUAGES`. ``None``/absent falls back to the default
    (``en``). A non-string or unknown value is a hard error so a typo fails fast
    rather than silently posting an English review the user asked to localize.
    """
    if raw is None:
        return DEFAULT_RESPONSE_LANGUAGE
    if not isinstance(raw, str):
        raise ConfigError(
            f"review.response_language must be a string, got {type(raw).__name__}"
        )
    canonical = _RESPONSE_LANGUAGE_ALIASES.get(raw.strip().lower())
    if canonical is None:
        raise ConfigError(
            f"review.response_language {raw!r} is not supported; allowed: "
            f"{RESPONSE_LANGUAGES} (aliases: "
            f"{sorted(_RESPONSE_LANGUAGE_ALIASES)})"
        )
    return canonical


def _parse_always_verify_categories(raw: Any) -> frozenset[str]:
    """Parse + validate ``review.always_verify_categories`` (trust-core lenses).

    Absent (``None``) falls back to the default trust-core set
    ({correctness, security}). An explicit empty list disables always-verify
    routing (only severity governs). Each entry must be a known category so a
    typo fails fast rather than silently re-opening the verify-strict leak.
    """
    if raw is None:
        return frozenset(DEFAULT_ALWAYS_VERIFY_CATEGORIES)
    if not _is_str_list(raw):
        raise ConfigError(
            "review.always_verify_categories must be a list of category strings"
        )
    for cat in raw:
        if cat not in LENSES:
            raise ConfigError(
                "review.always_verify_categories contains unknown category "
                f"{cat!r}; allowed: {LENSES}"
            )
    return frozenset(raw)


def _parse_unverified_confidence_gate(block: dict[str, Any], gate: float) -> float:
    """Parse + validate ``review.unverified_confidence_gate``.

    Findings that bypass the verifier must clear a HIGHER bar than the normal
    gate, else the FP leak this knob closes re-opens. When NOT set explicitly it
    defaults to ``max(DEFAULT_UNVERIFIED_CONFIDENCE_GATE, confidence_gate)`` so it
    is never below the gate (a high configured ``confidence_gate`` simply raises
    it). An EXPLICIT value below the gate, or outside ``[0, 1]``, is an error.
    """
    if "unverified_confidence_gate" not in block:
        return max(DEFAULT_UNVERIFIED_CONFIDENCE_GATE, gate)
    value = block.get("unverified_confidence_gate")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError("review.unverified_confidence_gate must be a number")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ConfigError("review.unverified_confidence_gate must be within [0, 1]")
    if value < gate:
        raise ConfigError(
            "review.unverified_confidence_gate "
            f"({value}) must be >= review.confidence_gate ({gate})"
        )
    return value


def _parse_lens_reasoning_effort(raw: Any) -> dict[str, str]:
    """Parse + validate ``review.lens_reasoning_effort`` (``{lens: effort}``).

    Each key must be a known lens and each value one of
    :data:`REASONING_EFFORTS`. An absent/empty map means every lens runs with
    reasoning OFF. Unknown lenses or invalid efforts fail fast (a typo here
    would silently degrade — or inflate the cost of — the finder pass).
    """
    if not isinstance(raw, Mapping):
        raise ConfigError("review.lens_reasoning_effort must be a mapping")
    out: dict[str, str] = {}
    for lens, effort in raw.items():
        if lens not in LENSES:
            raise ConfigError(
                "review.lens_reasoning_effort contains unknown lens "
                f"{lens!r}; allowed: {LENSES}"
            )
        if effort not in REASONING_EFFORTS:
            raise ConfigError(
                f"review.lens_reasoning_effort[{lens!r}] must be one of "
                f"{REASONING_EFFORTS}, got {effort!r}"
            )
        out[str(lens)] = str(effort)
    return out


def _parse_path_instructions(raw: Any) -> list[PathInstruction]:
    if not isinstance(raw, list):
        raise ConfigError("review.path_instructions must be a list")
    out: list[PathInstruction] = []
    for i, item in enumerate(raw):
        entry = _require_mapping(item, f"review.path_instructions[{i}]")
        if "path" not in entry or "instructions" not in entry:
            raise ConfigError(
                f"review.path_instructions[{i}] requires 'path' and 'instructions'"
            )
        out.append(
            PathInstruction(
                path=str(entry["path"]), instructions=str(entry["instructions"])
            )
        )
    return out


def _parse_model_roles(raw: Any) -> dict[str, ModelRole]:
    block = _require_mapping(raw, "model_roles")
    roles: dict[str, ModelRole] = {}
    for role, spec in block.items():
        entry = _require_mapping(spec, f"model_roles.{role}")
        if "model" not in entry or not entry["model"]:
            raise ConfigError(f"model_roles.{role} requires a non-empty 'model'")
        options = {k: v for k, v in entry.items() if k not in ("model", "region")}
        roles[str(role)] = ModelRole(
            model=str(entry["model"]),
            region=(str(entry["region"]) if entry.get("region") is not None else None),
            options=options,
        )
    return roles


def _parse_external_tools(raw: Any) -> ExternalTools:
    block = _require_mapping(raw, "external_tools")
    enabled = block.get("enabled", []) or []
    if not _is_str_list(enabled):
        raise ConfigError("external_tools.enabled must be a list of strings")
    return ExternalTools(enabled=list(enabled))


def _parse_telemetry(raw: Any) -> Telemetry:
    block = _require_mapping(raw, "telemetry")
    enabled = block.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("telemetry.enabled must be a boolean")
    mode = block.get("mode", "opt-out")
    if mode not in TELEMETRY_MODES:
        raise ConfigError(
            f"telemetry.mode must be one of {TELEMETRY_MODES}, got {mode!r}"
        )
    return Telemetry(enabled=enabled, mode=mode)


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)
