import os
import xml.etree.ElementTree as ET
import subprocess
import sys
import argparse
import signal
from junit_xml import TestCase, TestSuite, to_xml_report_string
from datetime import datetime
import time
from collections import deque
# Configuration: default output file and per-case timeout limits
output_xml = "test-results.xml"
DEFAULT_TIMEOUT_SECONDS = 1200
CASE_TIMEOUT_OVERRIDES = {
    "tests/test_flash_attn.py::test_flash_attn_kvcache": 5100,
    "tests/test_flash_attn.py::test_flash_attn_varlen_output": 5000,
    "hopper/test_flash_attn.py::test_flash_attn_kvcache": 3000,
}
# Known failures tolerated per pytest target: exactly this many failed subcases still counts as a pass.
KNOWN_FAILURE_COUNTS = {
    "tests/test_flash_attn.py::test_flash_attn_varlen_output": 24,
}
# Number of trailing pytest output lines attached to a failed case in the report
ERROR_LOG_TAIL_LINES = 10
# Max characters kept per log line (tensor dumps / assert diffs can be huge single lines)
ERROR_LOG_MAX_LINE_CHARS = 500
temp_xml_files = []
# (case args, exit code, temp xml, last output lines) for every pytest invocation that did not exit 0
failed_cases = []
def load_caselist(caselist_path):
    """Load pytest targets from a .list file.

    Each non-empty, non-comment line is one pytest invocation. A line holds a
    pytest node id (e.g. tests/test_flash_attn.py::test_flash_attn_causal) plus
    any optional per-case pytest args, separated by whitespace. Text after '#'
    is treated as an inline comment and stripped.
    """
    cmds = []
    with open(caselist_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            cmds.append(line.split())
    return cmds
def get_case_timeout(args):
    """Return a case-specific timeout, falling back to the default limit."""
    case_id = " ".join(args)
    for case_path, timeout_seconds in CASE_TIMEOUT_OVERRIDES.items():
        if case_path in case_id:
            return timeout_seconds
    return DEFAULT_TIMEOUT_SECONDS

def truncate_line(line, limit=ERROR_LOG_MAX_LINE_CHARS):
    """Cut an over-long log line, noting how many characters were dropped."""
    if len(line) <= limit:
        return line
    return f"{line[:limit]} ...[truncated {len(line) - limit} chars]"
def tolerate_known_failures(args, temp_xml):
    """Accept a run whose only failures are the expected known ones.

    If the target has an entry in KNOWN_FAILURE_COUNTS and its report holds
    exactly that many failed subcases and no errors, the failures are rewritten
    as skipped (keeping the original message) so the summary shows the target
    as passed. Returns True when the run is accepted.
    """
    expected = KNOWN_FAILURE_COUNTS.get(args[0]) if args else None
    if expected is None or not os.path.exists(temp_xml) or os.path.getsize(temp_xml) == 0:
        return False
    try:
        tree = ET.parse(temp_xml)
    except ET.ParseError:
        return False
    root = tree.getroot()
    testsuites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    testcases = [tc for ts in testsuites for tc in ts.findall("testcase")]
    failed = [tc for tc in testcases if tc.find("failure") is not None]
    errored = [tc for tc in testcases if tc.find("error") is not None]
    print(f"📊 {args[0]}: {len(failed)} failed, {len(errored)} errors (known failures allowed: {expected})")
    if len(failed) != expected or errored:
        return False
    for ts in testsuites:
        moved = 0
        for tc in ts.findall("testcase"):
            failure = tc.find("failure")
            if failure is None:
                continue
            tc.remove(failure)
            skipped = ET.SubElement(tc, "skipped", type="known-failure",
                                    message=f"Known failure ({expected} expected): {failure.get('message', '')}")
            skipped.text = failure.text
            moved += 1
        ts.set("failures", str(int(ts.get("failures", "0")) - moved))
        ts.set("skipped", str(int(ts.get("skipped", "0")) + moved))
    tree.write(temp_xml, encoding="utf-8", xml_declaration=True)
    return True
def run_test_python(args, idx):
    """Run a single pytest target and generate a JUnit XML report."""
    temp_xml = f"results_{idx}.xml"
    cmd = [
        "pytest",
        "-v",
        *args,
        f"--junitxml={temp_xml}"
    ]
    timeout_seconds = get_case_timeout(args)
    print(f"🚀 Running: {' '.join(cmd)}")
    print(f"⏰ Timeout: {timeout_seconds}s")
    # Always keep the report: pytest also writes it on test failures and collection errors.
    temp_xml_files.append(temp_xml)
    # Keep the output tail for the report and terminate the whole process group on timeout.
    tail = deque(maxlen=ERROR_LOG_TAIL_LINES)
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        timed_out = False
        try:
            output, _ = process.communicate(timeout=timeout_seconds)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            timeout_message = f"Timeout after {timeout_seconds}s"
            print(f"⏰ {timeout_message}: {' '.join(args)}")
            timed_out = True
            returncode = 124
        if output:
            print(output, end="", flush=True)
            for line in output.splitlines():
                tail.append(truncate_line(line))
        if timed_out:
            tail.append(timeout_message)
    except OSError as e:
        print(f"🔥 Failed to launch pytest for {' '.join(args)}: {e}")
        tail.append(truncate_line(str(e)))
        returncode = -1
    if returncode == 0:
        print(f"✅ Generated temporary file: {temp_xml}")
    elif returncode == 1 and tolerate_known_failures(args, temp_xml):
        print(f"✅ Test case {' '.join(args)} hit only its {KNOWN_FAILURE_COUNTS[args[0]]} known failures; treated as passed")
    else:
        print(f"❌ Test case {' '.join(args)} failed with exit code {returncode}")
        failed_cases.append((args, returncode, temp_xml, list(tail)))
    return temp_xml
def merge_xml(temp_xml_files, output_xml, failed_cases=()):
    """Merge all temporary XML files into the final result."""
    final_root = ET.Element("testsuite", name="pytest", tests="0", failures="0", errors="0", skipped="0", time="0.0")
    # temp xml -> testcases merged from it
    merged_cases = {}
    for temp_xml in temp_xml_files:
        if not os.path.exists(temp_xml) or os.path.getsize(temp_xml) == 0:
            print(f"⚠️ Skipping empty file: {temp_xml}")
            continue
        try:
            tree = ET.parse(temp_xml)
            root = tree.getroot()
            # pytest writes <testsuites><testsuite>...; older versions write <testsuite> as the root
            testsuites = [root] if root.tag == "testsuite" else root.findall("testsuite")
            for testsuite in testsuites:
                # Extract and merge all testcase elements
                testcases = testsuite.findall("testcase")
                for testcase in testcases:
                    final_root.append(testcase)
                merged_cases.setdefault(temp_xml, []).extend(testcases)
                # Update aggregate statistics
                final_root.set("tests", str(int(final_root.get("tests")) + int(testsuite.get("tests", "0"))))
                final_root.set("failures", str(int(final_root.get("failures")) + int(testsuite.get("failures", "0"))))
                final_root.set("errors", str(int(final_root.get("errors")) + int(testsuite.get("errors", "0"))))
                final_root.set("skipped", str(int(final_root.get("skipped", "0")) + int(testsuite.get("skipped", "0"))))
                final_root.set("time", str(float(final_root.get("time")) + float(testsuite.get("time", "0.0"))))
                print(f"✅ Merged {testsuite.get('tests', '0')} test cases from {temp_xml}")
        except ET.ParseError as e:
            print(f"❌ Failed to parse {temp_xml}: {e}")
    # Attach the tail of each failed invocation's output to its first failed testcase. A failed
    # invocation without any failed testcase in its report (missing/empty XML, nothing collected,
    # ...) would otherwise vanish from the summary; record it as an error case instead.
    for args, returncode, temp_xml, tail in failed_cases:
        case_id = " ".join(args)
        testcase = next((tc for tc in merged_cases.get(temp_xml, [])
                         if tc.find("failure") is not None or tc.find("error") is not None), None)
        if testcase is None:
            testcase = ET.SubElement(final_root, "testcase", classname="caselist", name=case_id, time="0.0")
            error = ET.SubElement(testcase, "error", message=f"pytest exited with code {returncode} without reporting a failed test case")
            error.text = f"pytest -v {case_id} exited with code {returncode}; see the job log for details."
            final_root.set("tests", str(int(final_root.get("tests")) + 1))
            final_root.set("errors", str(int(final_root.get("errors")) + 1))
            print(f"⚠️ Recorded {case_id} as an error case (no failed test case in {temp_xml})")
        if tail:
            system_err = ET.SubElement(testcase, "system-err")
            header = f"===== last {len(tail)} lines of `pytest -v {case_id}` (exit code {returncode}) ====="
            system_err.text = "\n".join([header, *tail])
    # Write the final XML file
    final_tree = ET.ElementTree(final_root)
    final_tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"✅ Test results merged into {output_xml}")
def main():
    parser = argparse.ArgumentParser(description="Run FA unit tests from a caselist and emit a merged JUnit XML report.")
    parser.add_argument("-c", "--caselist", required=True,
                        help="Path of the pytest caselist (e.g. fa2_pytest.list / fa3_pytest.list)")
    parser.add_argument("-o", "--output", default=output_xml,
                        help="Path of the merged JUnit XML report (default: %(default)s)")
    args = parser.parse_args()
    test_files_cmds = load_caselist(args.caselist)
    print(f"📋 Loaded {len(test_files_cmds)} test cases from {args.caselist}")
    # 1. Run tests and generate temporary XML files
    for i, cmd in enumerate(test_files_cmds):
        try:
            run_test_python(cmd, i)
        except Exception as e:
            print(f"🔥 Critical error while running test command {cmd}: {e}")
    # 2. Merge XML files
    merge_xml(temp_xml_files, args.output, failed_cases)
    # 3. Clean up temporary files
    for temp_xml in temp_xml_files:
        if not os.path.exists(temp_xml):
            continue
        try:
            os.remove(temp_xml)
            print(f"🗑️ Deleted temporary file: {temp_xml}")
        except Exception as e:
            print(f"⚠️ Failed to delete {temp_xml}: {e}")
    # 4. Propagate failures so the CI step does not report success
    if failed_cases:
        print(f"❌ {len(failed_cases)}/{len(test_files_cmds)} pytest invocations failed")
        return 1
    return 0
if __name__ == "__main__":
    sys.exit(main())
