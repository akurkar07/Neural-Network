import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import validate_model_data
from dependencies import Network
from randomiser import randomise_model


class NetworkTests(unittest.TestCase):
    structure = [4, 3, 2]

    def setUp(self):
        np.random.seed(7)
        self.model_data = randomise_model(self.structure)
        self.inputs = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
        self.targets = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def test_xavier_initialisation_is_reproducible(self):
        np.random.seed(42)
        first = randomise_model(self.structure)
        np.random.seed(42)
        second = randomise_model(self.structure)

        self.assertEqual(first, second)
        self.assertTrue(all(np.allclose(bias, 0.0) for bias in second["biases"]))

    def test_model_validation_rejects_wrong_weight_shape(self):
        invalid_data = copy.deepcopy(self.model_data)
        invalid_data["weights"][0].pop()

        self.assertFalse(validate_model_data(invalid_data, self.structure))

    def test_cpu_batch_update_changes_weights(self):
        network = Network(copy.deepcopy(self.model_data), self.structure, "numpy")
        original_weights = [weights.copy() for weights in network.weights]

        network.forwardBatch(self.inputs)
        network.backwardBatch(self.targets, 0.1)

        self.assertTrue(any(not np.array_equal(before, after) for before, after in zip(original_weights, network.weights)))

    @unittest.skipUnless(importlib.util.find_spec("cupy"), "CuPy is not installed")
    def test_cpu_and_gpu_forward_passes_agree(self):
        import cupy as cp

        cpu_network = Network(copy.deepcopy(self.model_data), self.structure, "numpy")
        gpu_network = Network(copy.deepcopy(self.model_data), self.structure, "cupy")

        cpu_output = cpu_network.forwardBatch(self.inputs)
        gpu_output = cp.asnumpy(gpu_network.forwardBatch(self.inputs))

        np.testing.assert_allclose(cpu_output, gpu_output, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()