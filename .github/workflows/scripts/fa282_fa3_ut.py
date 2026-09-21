import os
import xml.etree.ElementTree as ET
import subprocess
import sys
import argparse
from junit_xml import TestCase, TestSuite, to_xml_report_string
from datetime import datetime
import time
# Configuration: paths and output file
test_dir = "hopper"
output_xml = "test-results.xml"
# 显式指定要运行的测试文件（确保所有文件以 .py 结尾）
test_files_cmds = [
#    [os.path.join(test_dir, "test_flash_attn.py"), "test_flash_attn_output"], expect 2 failures
#    [os.path.join(test_dir, "test_flash_attn.py"), "test_flash_attn_varlen_output"], # expect 10 failures
    [os.path.join(test_dir, "test_flash_attn.py"), "test_flash_attn_kvcache"],
    [os.path.join(test_dir, "test_flash_attn.py"), "test_flash_attn_cluster"],
#    [os.path.join(test_dir, "test_flash_attn.py"), "test_flash_attn_race_condition"], # expect 60 failures
    [os.path.join(test_dir, "test_flash_attn.py"), "test_flash_attn_combine"],
]
temp_xml_files = []
def run_test_python(cmd, idx):
    """Run a single python test command and generate a JUnit XML report."""
    test_file, test_args = cmd[0], cmd[1]
    """运行单个测试文件并生成临时 XML"""
    temp_xml = f"results_{idx}.xml"
    cmd = [
        "pytest",
        "-v", "-s",
        f"{test_file}::{test_args}",
        f"--junitxml={temp_xml}"
    ]
    print(f"🚀 正在运行: {' '.join(cmd)}")
    print(f"文件路径: {test_file} | 存在: {os.path.exists(test_file)}")
    try:
        subprocess.run(cmd, check=True)
        temp_xml_files.append(temp_xml)
        print(f"✅ 已生成临时文件: {temp_xml}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试文件 {test_file} 执行失败: {e}")
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
    parser = argparse.ArgumentParser(description="Run FA unit tests and emit a merged JUnit XML report.")
    parser.add_argument("-o", "--output", default=output_xml,
                        help="Path of the merged JUnit XML report (default: %(default)s)")
    args = parser.parse_args()
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