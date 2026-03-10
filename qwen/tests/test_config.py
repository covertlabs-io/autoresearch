import unittest
from pathlib import Path

from qwen_geo.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_reads_prompt_and_label_style(self) -> None:
        config = load_config(Path("configs/geoguessr_baseline.toml"))
        self.assertEqual(config.prompt_style, "strict_json")
        self.assertEqual(config.label_style, "full_json")
        self.assertIn(576, config.search.image_sizes)

    def test_paths_resolve_from_project_root(self) -> None:
        config_path = Path("configs/smoke_local.toml").resolve()
        config = load_config(config_path)
        self.assertTrue(str(config.output_root(config_path)).endswith("/qwen/runs_smoke"))
        self.assertTrue(str(config.prepared_data_path(config_path)).endswith("/qwen/data/prepared_smoke"))


if __name__ == "__main__":
    unittest.main()
