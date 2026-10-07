"""No GPU: ensure failed CPU gates and duplicate slots cannot launch jobs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.radar_domain import coverage_pipeline as pipeline


class PipelineSafety(unittest.TestCase):
    def test_failed_CPU_gate_never_launches_and_retains_failure(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "launch"
            with patch.object(pipeline, "OUTPUT", output), \
                 patch.object(pipeline.evaluation.runtime, "identity"), \
                 patch.object(pipeline.evaluation, "load_protocol", return_value={"runtime_environment": {"expected": 1}}), \
                 patch.object(pipeline.evaluation, "environment", return_value={}), \
                 patch.object(pipeline.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "environment drift"):
                    pipeline.launch(Path("unused"), "sha")
            popen.assert_not_called()
            self.assertTrue((output / "failed.json").exists())

    def test_existing_pipeline_slot_rejects_duplicate_launch(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(pipeline, "OUTPUT", Path(d)), \
                 patch.object(pipeline.evaluation.runtime, "identity"), \
                 patch.object(pipeline.evaluation, "load_protocol", return_value={}), \
                 patch.object(pipeline.subprocess, "Popen") as popen:
                with self.assertRaises(FileExistsError):
                    pipeline.launch(Path("unused"), "sha")
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
