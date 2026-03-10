import tempfile
import unittest
from pathlib import Path

from qwen_geo.config import load_config
from qwen_geo.data import limit_split, load_prepared_dataset, prepare_dataset
from qwen_geo.smoke import evaluate_smoke_model, train_smoke_experiment


class SmokePipelineTests(unittest.TestCase):
    def test_smoke_backend_runs_end_to_end(self) -> None:
        config = load_config(Path("configs/smoke_local.toml"))
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            prepared = prepare_dataset(config, temp_root / "prepared", force=True)
            run_dir = temp_root / "run"
            run_dir.mkdir(parents=True, exist_ok=True)

            model, _, train_summary = train_smoke_experiment(config, prepared, run_dir)
            dataset_dict = load_prepared_dataset(prepared)
            eval_split = limit_split(dataset_dict["eval"], config.evaluation.eval_examples)
            metrics = evaluate_smoke_model(model, eval_split, config, run_dir)

            self.assertEqual(train_summary["backend"], "smoke")
            self.assertGreater(metrics["parse_rate"], 0.9)
            self.assertGreater(metrics["country_accuracy"], 0.9)
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "adapter" / "smoke_model.json").exists())


if __name__ == "__main__":
    unittest.main()
