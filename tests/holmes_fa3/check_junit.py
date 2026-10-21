import sys
import xml.etree.ElementTree as ET

cases = list(ET.parse(sys.argv[1]).getroot().iter("testcase"))
if len(cases) != 8 or any(
    c.find(tag) is not None for c in cases for tag in ("skipped", "failure", "error")
):
    raise SystemExit("FA3 ABI requires all 8 FP16/BF16 cases to execute and pass")
if any(not c.get("name", "").startswith("test_fa3_raw_graph[") for c in cases):
    raise SystemExit("Missing FA3 raw/graph coverage")
print("FA3 ABI JUnit: 8 executed, no failures/errors/skips")
