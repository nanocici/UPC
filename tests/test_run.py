"""入口的参数转发、失败停止、日志、结果保护和诊断保留检查。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
import run


def fake_success(command, log_path):
    Path(log_path).write_text("child completed\n", encoding="utf-8")
    output = Path(command[command.index("--output")+1])
    output.mkdir()
    if Path(command[1]).name == "inverse.py":
        data = [{"model": "F", "success": True}]
        (output/"example.json").write_text(json.dumps(data), encoding="utf-8")
    elif Path(command[1]).name == "uncertainty.py":
        data = [{"model": "F", "success": True, "message": "mode requires review",
                 "summary": {"direct_crossing": {"mean_yz_m": [2, 1], "mode_stable": False}}}]
        (output/"summary.json").write_text(json.dumps(data), encoding="utf-8")
    return 0


class RunTests(unittest.TestCase):
    def test_example_plan_forwards_settings_without_shell(self):
        steps = run.build_steps(Path("output with spaces"), model="both", samples=50,
                                starts=2, seed=8, drag=.3, sigma=.02, plot=True)
        self.assertEqual([name for name, command in steps], ["inverse", "uncertainty"])
        for name, command in steps:
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[command.index("--drag")+1], "0.3")
            self.assertEqual(command[command.index("--model")+1], "both")
            self.assertNotIn("--dataset", command)
        self.assertIn("--plot", steps[1][1])
        self.assertEqual(steps[1][1][steps[1][1].index("--samples")+1], "50")

    def test_development_plan_is_explicitly_development_only(self):
        steps = run.build_steps(Path("output"), task="development", dataset=Path("my data"))
        self.assertEqual([name for name, command in steps], ["inverse", "baselines", "evaluation"])
        for name, command in steps:
            self.assertEqual(command[command.index("--split")+1], "development")
        self.assertNotIn("uncertainty.py", " ".join(steps[-1][1]))

    def test_pipeline_keeps_review_flags_and_run_settings(self):
        with tempfile.TemporaryDirectory(prefix="run with spaces ") as temp:
            output = Path(temp)/"example with spaces"
            with patch.object(run, "execute_step", side_effect=fake_success):
                result = run.run_pipeline(output, samples=50, seed=8)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["diagnostics_status"], "review_required")
            self.assertFalse(result["models"][0]["mode_stable"])
            saved = json.loads((output/"run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["seed"], 8)
            self.assertEqual(len(result["steps"]), 2)
            self.assertTrue((output/"logs/01_inverse.log").exists())

    def test_first_failure_stops_later_steps_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"failed"
            def failed(command, log_path):
                Path(log_path).write_text("numerical child failed", encoding="utf-8")
                return 7
            with patch.object(run, "execute_step", side_effect=failed) as execute:
                result = run.run_pipeline(output)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(result["steps"][0]["returncode"], 7)
            self.assertTrue((output/"run_summary.json").exists())
            self.assertFalse((output/"uncertainty").exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            marker = output/"keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run.run_pipeline(output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_child_log_captures_both_streams_and_return_code(self):
        with tempfile.TemporaryDirectory(prefix="log with spaces ") as temp:
            path = Path(temp)/"child.log"
            code = "import sys; print('stdout'); print('stderr', file=sys.stderr); sys.exit(7)"
            returncode = run.execute_step([sys.executable, "-c", code], path)
            self.assertEqual(returncode, 7)
            self.assertIn("stdout", path.read_text(encoding="utf-8"))
            self.assertIn("stderr", path.read_text(encoding="utf-8"))

    def test_missing_child_output_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"missing"
            with patch.object(run, "execute_step", return_value=0):
                result = run.run_pipeline(output)
            self.assertEqual(result["status"], "failed")
            self.assertIn("输出汇总失败", result["message"])


if __name__ == "__main__":
    unittest.main()
