"""Tests for diff hunk-range parsing + valid comment-anchor positions (route.py).

These are pure/offline tests. The CRITICAL bug they guard: createReview comments
come verbatim from the model. If a (path, side, line) is outside the actual diff
GitHub 422s the ENTIRE batched review. route.py must therefore parse each
``@@ -a,b +c,d @@`` header into real line ranges and expose the exact set of
``(side, line)`` positions a comment may anchor on, so the adapter can drop or
clamp hallucinated positions before posting.
"""

from __future__ import annotations

from openrabbit.pipeline import route as route_mod

# A two-hunk file. New-side (RIGHT) lines 10,11,12 exist; 13 is context;
# the removed old-side (LEFT) line 12 exists. Anything else is NOT in the diff.
DIFF = """\
diff --git a/src/api/auth.py b/src/api/auth.py
index 1111111..2222222 100644
--- a/src/api/auth.py
+++ b/src/api/auth.py
@@ -10,4 +10,5 @@ def login(request):
 ctx_old_10
-removed_old_11
+added_new_11
+added_new_12
 ctx_both_13
@@ -40,2 +41,3 @@ def logout(request):
 ctx_old_40
+added_new_42
 ctx_both_43
"""


def _plan():
    return route_mod.route_diff(DIFF, lenses=["correctness"])


def _file_plan():
    plan = _plan()
    return next(f for f in plan.files if f.path == "src/api/auth.py")


class TestHunkRangeParsing:
    def test_hunk_header_ranges_are_parsed(self):
        fp = _file_plan()
        assert len(fp.hunks) == 2
        h0 = fp.hunks[0]
        # @@ -10,4 +10,5 @@
        assert h0.old_start == 10
        assert h0.old_count == 4
        assert h0.new_start == 10
        assert h0.new_count == 5
        h1 = fp.hunks[1]
        # @@ -40,2 +41,3 @@
        assert h1.old_start == 40
        assert h1.new_start == 41
        assert h1.new_count == 3

    def test_hunk_header_without_counts_defaults_to_one(self):
        # "@@ -5 +5 @@" means a single-line range on both sides (count omitted).
        diff = (
            "diff --git a/x.py b/x.py\n"
            "--- a/x.py\n+++ b/x.py\n"
            "@@ -5 +5 @@\n-old\n+new\n"
        )
        plan = route_mod.route_diff(diff, lenses=["correctness"])
        h = plan.files[0].hunks[0]
        assert h.old_start == 5
        assert h.old_count == 1
        assert h.new_start == 5
        assert h.new_count == 1


class TestValidAnchorPositions:
    def test_right_side_added_and_context_lines_are_valid(self):
        fp = _file_plan()
        valid = route_mod.valid_anchor_positions(fp)
        # New-side numbering starts at 10. Lines: 10 ctx, 11 added, 12 added,
        # 13 ctx (hunk1); 41 ctx, 42 added, 43 ctx (hunk2).
        assert ("RIGHT", 10) in valid
        assert ("RIGHT", 11) in valid
        assert ("RIGHT", 12) in valid
        assert ("RIGHT", 13) in valid
        assert ("RIGHT", 42) in valid
        assert ("RIGHT", 43) in valid

    def test_left_side_removed_and_context_lines_are_valid(self):
        fp = _file_plan()
        valid = route_mod.valid_anchor_positions(fp)
        # Old-side numbering starts at 10: 10 ctx, 11 removed, 12 ctx.
        assert ("LEFT", 10) in valid
        assert ("LEFT", 11) in valid

    def test_lines_outside_any_hunk_are_invalid(self):
        fp = _file_plan()
        valid = route_mod.valid_anchor_positions(fp)
        # A line far outside both hunks (hallucinated) must NOT be valid.
        assert ("RIGHT", 999) not in valid
        assert ("RIGHT", 25) not in valid  # gap between the two hunks
        assert ("LEFT", 500) not in valid

    def test_added_line_has_no_left_position(self):
        # An added line only advances the new-side counter, so it must not show
        # up as a LEFT (old-side) anchor.
        fp = _file_plan()
        valid = route_mod.valid_anchor_positions(fp)
        # new line 11 was an addition; old-side 11 was the removed line, not 12.
        assert ("LEFT", 12) in valid  # context line on old side
        # the second hunk's added new-42 has no matching old-side line number
        assert ("LEFT", 42) not in valid

    def test_valid_positions_by_file_keys_on_path(self):
        plan = _plan()
        by_file = route_mod.valid_positions_by_file(plan)
        assert "src/api/auth.py" in by_file
        assert ("RIGHT", 11) in by_file["src/api/auth.py"]
