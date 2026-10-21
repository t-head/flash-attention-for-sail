"""Require actual execution, including graph cases, rather than all-skipped green."""

import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
cases = list(root.iter("testcase"))
if not cases:
    raise SystemExit("No tests collected")
if any(c.find(tag) is not None for c in cases for tag in ("failure", "error", "skipped")):
    raise SystemExit("Failed, erroneous or skipped ABI tests")
for prefix, minimum in {
    "test_workspace_graph[": 36,
    "test_invalid_layout[": 8,
    "test_query_errors": 1,
}.items():
    if sum(c.get("name", "").startswith(prefix) for c in cases) < minimum:
        raise SystemExit(f"Missing ABI coverage: {prefix}")
print(f"ABI JUnit: {len(cases)} executed, no failures/errors/skips")
