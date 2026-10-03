"""NVML may report idle, low-memory GPUs that still require a driver reset."""
import unittest
from unittest.mock import patch

from experiments.agent_feedback.initial_runs import require_idle_gpus


def nvml_responses(recovery="None", processes="[N/A]\n" * 4):
    health = "<nvidia_smi_log>" + "".join(
        f'<gpu id="00000000:0{i}:00.0"><gpu_recovery_action>{recovery}</gpu_recovery_action>'
        '<fb_memory_usage><used>3 MiB</used></fb_memory_usage><processes/></gpu>'
        for i in range(4)) + "</nvidia_smi_log>"
    return {
        ("nvidia-smi", "-q", "-x"): health,
        ("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"): processes,
        ("nvidia-smi",): "No running processes found\n",
        ("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"): "3\n" * 4,
    }


class GPUReadinessTest(unittest.TestCase):
    def mock_nvml(self, responses):
        # No subprocess or server command can run in these regression tests.
        return patch("experiments.agent_feedback.initial_runs.subprocess.check_output",
                     side_effect=lambda command, **kwargs: responses[tuple(command)])

    def test_reset_required_rejected_even_with_three_mib_and_no_processes(self):
        with self.mock_nvml(nvml_responses(recovery="Reset")) as query:
            with self.assertRaisesRegex(RuntimeError, "required recovery action.*Reset"):
                require_idle_gpus()
            # Reject the driver fault before idle-looking process/memory data
            # can be mistaken for evidence that CUDA is usable.
            query.assert_called_once_with(["nvidia-smi", "-q", "-x"], text=True)

    def test_healthy_but_busy_device_is_rejected(self):
        with self.mock_nvml(nvml_responses(processes="12345\n")):
            with self.assertRaisesRegex(RuntimeError, "GPU processes are present"):
                require_idle_gpus()

    def test_healthy_idle_devices_with_na_pid_rows_pass(self):
        with self.mock_nvml(nvml_responses()) as query:
            self.assertIsNone(require_idle_gpus())
            self.assertEqual(query.call_count, 4)


if __name__ == "__main__":
    unittest.main()
