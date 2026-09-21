#!/usr/bin/env python3
"""Parse JUnit XML test results and generate a Markdown summary.

Usage:
    python3 display_test_summary.py <path-to-test-results.xml>

If the environment variable GITHUB_STEP_SUMMARY is set, the Markdown
summary is appended to that file (for GitHub Actions).  Otherwise it is
printed to stdout, which is useful for local debugging.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET


def parse_test_results(xml_path):
    """Parse a JUnit XML file and return a dict with summary statistics
    plus a list of per-testcase dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    total = int(root.get("tests", 0))
    failures = int(root.get("failures", 0))
    errors = int(root.get("errors", 0))
    skipped = int(root.get("skipped", 0))
    passed = total - failures - errors - skipped
    time = float(root.get("time", 0))

    testcases = []
    for tc in root.findall(".//testcase"):
        name = tc.get("name", "unknown")
        t = float(tc.get("time", 0))
        is_skipped = tc.find("skipped") is not None
        is_failure = tc.find("failure") is not None
        is_error = tc.find("error") is not None
        testcases.append({
            "name": name,
            "time": t,
            "passed": not is_failure and not is_error and not is_skipped,
            "skipped": is_skipped,
        })

    return {
        "total": total,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": passed,
        "time": time,
        "testcases": testcases,
    }


def group_by_parent(testcases):
    """Collapse parametrized subcases into their parent case.

    pytest names parametrized cases like ``test_foo[param]``; the parent is the
    part before ``[``.  A parent's status is Failed if any subcase failed or
    errored, else Skipped if every subcase was skipped, else Passed.  Subcase
    times are summed and the subcase count is recorded.
    """
    groups = {}
    order = []
    for tc in testcases:
        parent = tc["name"].split("[", 1)[0]
        g = groups.get(parent)
        if g is None:
            g = {"name": parent, "time": 0.0, "count": 0,
                 "has_fail": False, "non_skipped": 0}
            groups[parent] = g
            order.append(parent)
        g["time"] += tc["time"]
        g["count"] += 1
        if not tc["passed"] and not tc["skipped"]:
            g["has_fail"] = True
        if not tc["skipped"]:
            g["non_skipped"] += 1

    result = []
    for parent in order:
        g = groups[parent]
        failed = g["has_fail"]
        all_skipped = g["non_skipped"] == 0
        result.append({
            "name": g["name"],
            "time": g["time"],
            "count": g["count"],
            "passed": (not failed) and (not all_skipped),
            "skipped": all_skipped and not failed,
        })
    return result


def generate_markdown_summary(stats, show_subcases=False):
    """Render the parsed statistics as a Markdown string.

    Uses an HTML table layout for each section.  All inner content is pure
    HTML because GitHub Flavored Markdown does not parse Markdown syntax
    (e.g. pipe tables) inside an HTML block.
    """
    lines = []
    run_number = os.environ.get("GITHUB_RUN_NUMBER")
    repo = os.environ.get("GITHUB_REPOSITORY", "").split("/")[-1]
    branch = os.environ.get("GITHUB_REF_NAME")

    # Build title: <Repo> Test Results #<run-number> @ <branch>
    title_parts = []
    if run_number:
        title_parts.append(f"#{run_number}")
    if branch:
        title_parts.append(f"@ {branch}")

    suffix = f" {' '.join(title_parts)}" if title_parts else ""
    repo_prefix = f"{repo} " if repo else ""
    lines.append(f"## {repo_prefix}Test Results{suffix}\n")
    # ---- Summary overview ----
    lines.append('<h3>概览</h3>')
    lines.append('<table>')
    lines.append('<tr>'
                 '<th>Total Cases</th>'
                 '<th>✅ Passed</th>'
                 '<th>❌ Failed</th>'
                 '<th>⏭ Skipped</th>'
                 '<th>⏱ Run Time</th>'
                 '</tr>')
    lines.append('<tr>'
                 f"<td>{stats['total']}</td>"
                 f"<td>{stats['passed']}</td>"
                 f"<td>{stats['failures'] + stats['errors']}</td>"
                 f"<td>{stats['skipped']}</td>"
                 f"<td>{stats['time']:.1f}s</td>"
                 '</tr>')
    lines.append('</table>')
    # ---- Test-case details (below overview, collapsed by default) ----
    if show_subcases:
        rows = stats["testcases"]
        case_header = "Test Case"
    else:
        rows = group_by_parent(stats["testcases"])
        case_header = "Test Case"
    lines.append('<details>')
    lines.append('<summary>Test Details</summary>')
    lines.append('<table>')
    lines.append(f'<tr><th>{case_header}</th><th>Time</th><th>Status</th></tr>')
    for tc in rows:
        if tc.get("skipped"):
            status = "⏭ Skipped"
        elif tc["passed"]:
            status = "✅ Passed"
        else:
            status = "❌ Failed"
        name = tc["name"]
        if not show_subcases and tc.get("count", 1) > 1:
            name = f"{name} ({tc['count']} subcases)"
        lines.append(f"<tr><td>{name}</td><td>{tc['time']:.1f}s</td><td>{status}</td></tr>")
    lines.append('</table>')
    lines.append('</details>')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown summary from JUnit XML test results."
    )
    parser.add_argument(
        "xml_file",
        help="Path to the test-results.xml file",
    )
    parser.add_argument(
        "--show-subcases",
        action="store_true",
        help=(
            "List every parametrized subcase (e.g. name[param]).  By default "
            "only parent cases are summarized, with subcases collapsed."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.xml_file):
        print(f"❌ File not found: {args.xml_file}", file=sys.stderr)
        sys.exit(1)

    try:
        stats = parse_test_results(args.xml_file)
    except ET.ParseError as e:
        print(f"❌ Failed to parse XML: {e}", file=sys.stderr)
        sys.exit(1)

    markdown = generate_markdown_summary(stats, show_subcases=args.show_subcases)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(markdown)
        print(f"✅ Summary written to {summary_path}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
