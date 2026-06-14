"""Tests for the walkthrough enrichment (SPEC section 6, step 6).

``build_walkthrough(pr_context, file_plans, findings) -> markdown`` produces the
sticky-walkthrough body: a high-level summary, a GROUPED changed-files table, an
optional Mermaid diagram (only for interaction-flavored changes), and the
findings summary table that emit already renders today.

Every test is fully offline & deterministic: no network, no model calls, no
credentials. ``FilePlan``/``Finding`` are built directly.
"""

from __future__ import annotations

from openrabbit.findings import Finding, compute_fingerprint
from openrabbit.pipeline import walkthrough as wt
from openrabbit.pipeline.route import FilePlan, Hunk


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #
def _file_plan(
    path: str,
    *,
    file_type: str = "code",
    risk: str = "medium",
    lenses: list[str] | None = None,
    diff: str = "",
) -> FilePlan:
    hunks = [Hunk(header="@@ -1 +1 @@", text=diff)] if diff else []
    return FilePlan(
        path=path,
        file_type=file_type,
        risk=risk,
        lenses=lenses if lenses is not None else ["correctness"],
        model_role="finder",
        hunks=hunks,
    )


def _finding(
    file: str = "src/api/auth.py",
    *,
    title: str = "SQL injection",
    severity: str = "high",
    category: str = "security",
    rule: str = "openrabbit/security/sqli",
) -> Finding:
    fp = compute_fingerprint(file, rule, title)
    return Finding(
        file=file,
        start_line=12,
        end_line=14,
        side="RIGHT",
        severity=severity,
        category=category,
        confidence=0.95,
        title=title,
        body="Rationale.",
        rule_id=rule,
        fingerprint=fp,
    )


# --------------------------------------------------------------------------- #
# summary                                                                       #
# --------------------------------------------------------------------------- #
class TestSummary:
    def test_summary_present(self):
        plans = [_file_plan("src/api/auth.py"), _file_plan("src/api/users.py")]
        md = wt.build_walkthrough({}, plans, [_finding()])
        assert "## Walkthrough" in md
        # A high-level summary sentence exists before the table.
        summary = md.split("|", 1)[0]
        assert len(summary.strip()) > 0
        # 2-3 sentences (bounded): count terminal punctuation.
        sentence_ct = summary.count(".") + summary.count("!") + summary.count("?")
        assert 1 <= sentence_ct <= 4

    def test_summary_reflects_file_count(self):
        plans = [_file_plan("a.py"), _file_plan("b.py"), _file_plan("c.py")]
        md = wt.build_walkthrough({}, plans, [])
        assert "3" in md  # mentions the number of changed files

    def test_uses_pr_title_when_present(self):
        plans = [_file_plan("src/api/auth.py")]
        md = wt.build_walkthrough({"title": "Add OAuth token refresh"}, plans, [])
        assert "Add OAuth token refresh" in md

    def test_pr_title_is_escaped_not_executed(self):
        # PR title is UNTRUSTED data; markdown control chars must be neutralized
        # (pipes escaped) so it can never break out of a table cell downstream.
        plans = [_file_plan("src/api/auth.py")]
        md = wt.build_walkthrough({"title": "evil | title <!-- x -->"}, plans, [])
        # The raw unescaped pipe sequence must not appear verbatim in a way that
        # could corrupt a table row.
        assert "evil \\| title" in md


# --------------------------------------------------------------------------- #
# grouped changed-files table                                                   #
# --------------------------------------------------------------------------- #
class TestGroupedTable:
    def test_groups_related_files_into_one_row(self):
        # Two files in the same directory -> ONE grouped row, not two.
        plans = [
            _file_plan("src/api/auth.py"),
            _file_plan("src/api/users.py"),
        ]
        md = wt.build_walkthrough({}, plans, [])
        # The shared group key (directory) appears as a row.
        assert "src/api" in md
        # Both files are represented within that group.
        assert "auth.py" in md
        assert "users.py" in md
        # Only one data row for the api group (header rows + 1 data row).
        data_rows = [
            ln for ln in md.splitlines() if ln.startswith("|") and "src/api" in ln
        ]
        assert len(data_rows) == 1

    def test_separate_directories_become_separate_rows(self):
        plans = [
            _file_plan("src/api/auth.py"),
            _file_plan("docs/guide.md", file_type="docs", lenses=[]),
            _file_plan("tests/test_auth.py", file_type="test"),
        ]
        md = wt.build_walkthrough({}, plans, [])
        rows = [ln for ln in md.splitlines() if ln.startswith("| `")]
        # api, docs, tests -> 3 distinct groups.
        assert len(rows) >= 3

    def test_table_has_description_and_purpose_columns(self):
        plans = [_file_plan("tests/test_auth.py", file_type="test")]
        md = wt.build_walkthrough({}, plans, [])
        # Header carries plain-language description + change-purpose columns.
        header = next(
            ln
            for ln in md.splitlines()
            if ln.startswith("|")
            and "---" not in ln
            and ("Files" in ln or "Group" in ln or "Change" in ln)
        )
        lowered = header.lower()
        assert "files" in lowered or "group" in lowered
        assert "change" in lowered or "summary" in lowered or "description" in lowered

    def test_description_is_plain_language_by_type(self):
        plans = [
            _file_plan("tests/test_auth.py", file_type="test"),
            _file_plan("docs/readme.md", file_type="docs", lenses=[]),
        ]
        md = wt.build_walkthrough({}, plans, [])
        lowered = md.lower()
        # Heuristic plain-language descriptions inferred from file type/path.
        assert "test" in lowered
        assert "doc" in lowered


# --------------------------------------------------------------------------- #
# Mermaid diagram (interaction-only)                                            #
# --------------------------------------------------------------------------- #
class TestMermaid:
    def test_mermaid_present_for_interaction_change(self):
        # API-flavored fileset (api handler + service + client) -> diagram.
        plans = [
            _file_plan(
                "src/api/orders.py",
                file_type="code",
                diff="+def create_order(req):\n+    return service.place(req)",
            ),
            _file_plan(
                "src/services/order_service.py",
                file_type="code",
                diff="+def place(order):\n+    requests.post(url, json=order)",
            ),
        ]
        md = wt.build_walkthrough({}, plans, [])
        assert "```mermaid" in md
        # _render_mermaid deterministically emits a `flowchart LR`; pin that exact
        # contract rather than accepting any diagram kind.
        assert "flowchart LR" in md

    def test_mermaid_omitted_for_docs_only_change(self):
        plans = [
            _file_plan("docs/intro.md", file_type="docs", lenses=[]),
            _file_plan("README.md", file_type="docs", lenses=[]),
        ]
        md = wt.build_walkthrough({}, plans, [])
        assert "```mermaid" not in md

    def test_mermaid_omitted_for_simple_single_file_change(self):
        # A lone, non-interaction code tweak shouldn't manufacture a diagram.
        plans = [
            _file_plan(
                "src/utils/strings.py",
                file_type="code",
                diff="+def upper(s):\n+    return s.upper()",
            )
        ]
        md = wt.build_walkthrough({}, plans, [])
        assert "```mermaid" not in md

    def test_mermaid_for_async_event_flow(self):
        plans = [
            _file_plan(
                "src/workers/consumer.py",
                file_type="code",
                diff="+async def on_event(msg):\n+    await queue.publish(msg)",
            ),
            _file_plan(
                "src/workers/producer.py",
                file_type="code",
                diff="+async def emit(evt):\n+    await broker.send(evt)",
            ),
        ]
        md = wt.build_walkthrough({}, plans, [])
        assert "```mermaid" in md


# --------------------------------------------------------------------------- #
# findings table + boundedness                                                  #
# --------------------------------------------------------------------------- #
class TestFindingsAndBounds:
    def test_includes_findings_table(self):
        plans = [_file_plan("src/api/auth.py")]
        md = wt.build_walkthrough({}, plans, [_finding()])
        # The findings summary table that emit renders today is embedded.
        assert "SQL injection" in md
        assert "Severity" in md or "severity" in md.lower()

    def test_no_findings_still_renders_walkthrough(self):
        plans = [_file_plan("src/api/auth.py")]
        md = wt.build_walkthrough({}, plans, [])
        assert "## Walkthrough" in md
        # Findings section communicates the clean result.
        assert "No issues" in md or "no issues" in md.lower()

    def test_output_is_bounded(self):
        # Many files must not produce an unbounded table.
        plans = [_file_plan(f"src/mod{i}/file{i}.py") for i in range(200)]
        md = wt.build_walkthrough({}, plans, [])
        rows = [ln for ln in md.splitlines() if ln.startswith("| `")]
        assert len(rows) <= wt.MAX_TABLE_ROWS
        # A truncation note is shown when groups exceed the cap.
        assert "more" in md.lower()

    def test_deterministic(self):
        plans = [
            _file_plan("src/api/auth.py"),
            _file_plan("src/api/users.py"),
            _file_plan("tests/test_auth.py", file_type="test"),
        ]
        a = wt.build_walkthrough({"title": "X"}, plans, [_finding()])
        b = wt.build_walkthrough({"title": "X"}, plans, [_finding()])
        assert a == b

    def test_empty_fileset_is_safe(self):
        md = wt.build_walkthrough({}, [], [])
        assert isinstance(md, str)
        assert "## Walkthrough" in md


# --------------------------------------------------------------------------- #
# heuristic edge cases (purpose / truncation / mermaid labels)                  #
# --------------------------------------------------------------------------- #
class TestHeuristicEdges:
    def test_long_title_is_truncated(self):
        long_title = "x" * 500
        md = wt.build_walkthrough({"title": long_title}, [_file_plan("a.py")], [])
        assert "…" in md
        # The full 500-char title must not appear verbatim.
        assert long_title not in md

    def test_purpose_removes_for_deletion_only(self):
        plan = _file_plan(
            "src/dead.py", file_type="code", diff="-def gone():\n-    pass"
        )
        md = wt.build_walkthrough({}, [plan], [])
        assert "Removes" in md

    def test_purpose_updates_for_mixed_changes(self):
        plan = _file_plan("src/mix.py", file_type="code", diff="-old = 1\n+new = 2")
        md = wt.build_walkthrough({}, [plan], [])
        assert "Updates" in md

    def test_many_files_in_one_group_truncates_filenames(self):
        plans = [_file_plan(f"src/api/f{i}.py") for i in range(20)]
        md = wt.build_walkthrough({}, plans, [])
        # Single group (src/api) with a "+N more" filename suffix.
        api_row = next(ln for ln in md.splitlines() if ln.startswith("| `src/api`"))
        assert "more" in api_row

    def test_mermaid_disambiguates_same_filename(self):
        # Same basename in two interaction dirs -> diagram still renders with
        # distinct nodes (no crash, no duplicate identifiers).
        plans = [
            _file_plan(
                "src/api/handler.py",
                diff="+def h(req):\n+    return service.call(req)",
            ),
            _file_plan(
                "src/workers/handler.py",
                diff="+async def h(msg):\n+    await queue.publish(msg)",
            ),
        ]
        md = wt.build_walkthrough({}, plans, [])
        assert "```mermaid" in md
        assert "flowchart LR" in md

    def test_mermaid_label_escapes_double_quote_in_filename(self):
        # An UNTRUSTED changed-file path containing a double-quote must not close
        # the Mermaid label early (which would corrupt the diagram block).
        plans = [
            _file_plan(
                'src/api/ev"il.py',
                diff="+def h(req):\n+    return service.call(req)",
            ),
            _file_plan(
                "src/services/order_service.py",
                diff="+def place(o):\n+    requests.post(url, json=o)",
            ),
        ]
        md = wt.build_walkthrough({}, plans, [])
        assert "```mermaid" in md
        # In the Mermaid `["..."]` label a bare quote would terminate the string
        # early and corrupt the diagram, so it must be the Mermaid entity instead.
        node_lines = [
            ln for ln in md.splitlines() if ln.strip().endswith('"]') and "[" in ln
        ]
        assert node_lines, "expected at least one mermaid node label line"
        assert any("&quot;" in ln for ln in node_lines)
        # No node label line carries a raw double-quote inside its display text.
        for ln in node_lines:
            inner = ln[ln.index('["') + 2 : ln.rindex('"]')]
            assert '"' not in inner

    def test_mermaid_omitted_for_pure_async_helpers(self):
        # Two pure-compute async helpers (no network/IPC verb, no interaction
        # path) must NOT manufacture an interaction diagram (SPEC low-noise).
        plans = [
            _file_plan(
                "src/mathlib/adder.py",
                file_type="code",
                diff="+async def add(x, y):\n+    return x + y",
            ),
            _file_plan(
                "src/mathlib/multiplier.py",
                file_type="code",
                diff="+async def mul(x, y):\n+    return x * y",
            ),
        ]
        md = wt.build_walkthrough({}, plans, [])
        assert "```mermaid" not in md


# --------------------------------------------------------------------------- #
# emit wiring: the sticky walkthrough now carries the grouped table             #
# --------------------------------------------------------------------------- #
_SAMPLE_DIFF = """\
diff --git a/src/api/auth.py b/src/api/auth.py
index 1111111..2222222 100644
--- a/src/api/auth.py
+++ b/src/api/auth.py
@@ -10,6 +10,9 @@ def login(request):
     user = lookup(request.user)
     token = request.GET["token"]
-    if token == user.token:
+    query = "SELECT * FROM users WHERE token = '" + token + "'"
+    db.execute(query)
+    if token == user.token:
         return ok()
     return deny()
diff --git a/src/api/users.py b/src/api/users.py
index 3333333..4444444 100644
--- a/src/api/users.py
+++ b/src/api/users.py
@@ -1,2 +1,3 @@
 import x
+def fetch(uid):
+    return requests.get(url)
"""


class TestEmitWiring:
    """The orchestrator's emit step must surface the enriched walkthrough
    (grouped changed-files table) in the sticky walkthrough payload — not just
    the minimal summary it used before."""

    def _config(self):
        from openrabbit.config import load_config

        return load_config(
            {
                "version": 1,
                "review": {
                    "profile": "balanced",
                    "confidence_gate": 0.80,
                    "lenses": ["correctness", "security"],
                },
                "model_roles": {
                    "finder": {
                        "model": "amazon.nova-pro-v1:0",
                        "region": "ap-northeast-2",
                    },
                    "verifier": {"model": "openai.gpt-5.5", "region": "us-east-2"},
                },
            }
        )

    def _emit_findings_result(self, findings):
        from openrabbit.domain import (
            CompletionResult,
            FinishReason,
            ToolCall,
            Usage,
        )

        return CompletionResult(
            text="",
            tool_calls=[
                ToolCall(id="c", name="emit_findings", args={"findings": findings})
            ],
            finish_reason=FinishReason.TOOL_USE,
            usage=Usage(input_tokens=10, output_tokens=5),
        )

    def _verify_batch_result(self, verdicts):
        """A batched verify result: ``verdicts`` is a list of (id, keep, conf)."""
        from openrabbit.domain import (
            CompletionResult,
            FinishReason,
            ToolCall,
            Usage,
        )

        return CompletionResult(
            text="",
            tool_calls=[
                ToolCall(
                    id="v",
                    name="verify_findings",
                    args={
                        "verdicts": [
                            {
                                "id": vid,
                                "keep": keep,
                                "confidence": conf,
                                "rationale": "ok",
                            }
                            for vid, keep, conf in verdicts
                        ]
                    },
                )
            ],
            finish_reason=FinishReason.TOOL_USE,
            usage=Usage(input_tokens=8, output_tokens=2),
        )

    def test_sticky_walkthrough_contains_grouped_table(self):
        from openrabbit.pipeline import orchestrator as orch
        from openrabbit.providers.base import FakeProvider

        config = self._config()
        finder = FakeProvider(
            [
                self._emit_findings_result(
                    [
                        {
                            "file": "src/api/auth.py",
                            "startLine": 12,
                            "endLine": 14,
                            "side": "RIGHT",
                            "severity": "high",
                            "category": "correctness",
                            "confidence": 70,
                            "title": "correctness issue",
                            "body": "b",
                            "ruleId": "openrabbit/correctness/x",
                        }
                    ]
                ),
                self._emit_findings_result(
                    [
                        {
                            "file": "src/api/auth.py",
                            "startLine": 12,
                            "endLine": 14,
                            "side": "RIGHT",
                            "severity": "high",
                            "category": "security",
                            "confidence": 95,
                            "title": "SQL injection",
                            "body": "b",
                            "ruleId": "openrabbit/security/sqli",
                        }
                    ]
                ),
                # users.py also routes to two lenses.
                self._emit_findings_result([]),
                self._emit_findings_result([]),
            ]
        )
        verifier = FakeProvider(
            [self._verify_batch_result([(0, True, 0.95), (1, True, 0.92)])]
        )

        result = orch.review(
            config,
            pr_context={
                "draft": False,
                "state": "open",
                "head_sha": "abc",
                "diff": _SAMPLE_DIFF,
                "title": "Harden auth",
                "body": "PR body",
            },
            providers={"finder": finder, "verifier": verifier},
        )
        assert result.reviewed is True
        sticky = result.emitted["sticky_walkthrough"]
        # The enriched walkthrough now carries the grouped changed-files table.
        assert "## Walkthrough" in sticky
        assert "### Changed files" in sticky
        assert "| Group | Files | Change summary |" in sticky
        assert "src/api" in sticky
        # ...and still embeds the findings table.
        assert "SQL injection" in sticky
