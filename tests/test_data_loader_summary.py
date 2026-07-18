import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

import config
from data_loader import DataLoader


class DataLoaderSummaryTests(unittest.TestCase):
    def test_hr_summary_reports_synthetic_source_without_legacy_csv_name(self):
        loader = DataLoader(csv_path="DRG3_MdFoF.csv")
        loader.time = np.array([0.0, 0.1, 0.2])
        loader.neuron_names = ["hr_x", "hr_y", "hr_z"]
        loader.n_samples = 3
        loader.n_neurons = 3
        loader.spike_indices = {"hr_x": np.array([1])}

        output = io.StringIO()
        with mock.patch.object(config, "DATASET_MODE", "hr"):
            with redirect_stdout(output):
                loader.summary()

        summary = output.getvalue()
        self.assertIn(
            "Synthetic Hindmarsh-Rose generated trajectory (RK4)", summary
        )
        self.assertNotIn("DRG3_MdFoF.csv", summary)


if __name__ == "__main__":
    unittest.main()
