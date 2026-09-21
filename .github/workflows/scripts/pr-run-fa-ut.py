import os
import xml.etree.ElementTree as ET
import subprocess
import sys
import argparse
from junit_xml import TestCase, TestSuite, to_xml_report_string
from datetime import datetime
import time
# Configuration: default output file
output_xml = "test-results.xml"
temp_xml_files = []
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
def run_test_python(args, idx):
    """Run a single pytest target and generate a JUnit XML report."""
    """运行单个测试用例并生成临时 XML"""
    temp_xml = f"results_{idx}.xml"
    cmd = [
        "pytest",
        "-v",
        *args,
        f"--junitxml={temp_xml}"
    ]
    print(f"🚀 正在运行: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        temp_xml_files.append(temp_xml)
        print(f"✅ 已生成临时文件: {temp_xml}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试用例 {' '.join(args)} 执行失败: {e}")
    return temp_xml
def merge_xml(temp_xml_files, output_xml):
    """Merge all temporary XML files into the final result."""
    final_root = ET.Element("testsuite", name="pytest", tests="0", failures="0", errors="0", skipped="0", time="0.0")
    for temp_xml in temp_xml_files:
        if not os.path.exists(temp_xml) or os.path.getsize(temp_xml) == 0:
            print(f"⚠️ Skipping empty file: {temp_xml}")
            continue
        try:
            tree = ET.parse(temp_xml)
            root = tree.getroot()
            # Find <testsuite> elements under <testsuites>
            for testsuite in root.findall("testsuite"):
                # Extract and merge all testcase elements
                for testcase in testsuite.findall("testcase"):
                    final_root.append(testcase)
                # Update aggregate statistics
                final_root.set("tests", str(int(final_root.get("tests")) + int(testsuite.get("tests", "0"))))
                final_root.set("failures", str(int(final_root.get("failures")) + int(testsuite.get("failures", "0"))))
                final_root.set("errors", str(int(final_root.get("errors")) + int(testsuite.get("errors", "0"))))
                final_root.set("skipped", str(int(final_root.get("skipped", "0")) + int(testsuite.get("skipped", "0"))))
                final_root.set("time", str(float(final_root.get("time")) + float(testsuite.get("time", "0.0"))))
                print(f"✅ Merged {testsuite.get('tests', '0')} test cases from {temp_xml}")
        except ET.ParseError as e:
            print(f"❌ Failed to parse {temp_xml}: {e}")
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
    merge_xml(temp_xml_files, args.output)
    # 3. Clean up temporary files
    for temp_xml in temp_xml_files:
        try:
            os.remove(temp_xml)
            print(f"🗑️ Deleted temporary file: {temp_xml}")
        except Exception as e:
            print(f"⚠️ Failed to delete {temp_xml}: {e}")
if __name__ == "__main__":
    main()
