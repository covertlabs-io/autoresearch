import unittest
from pathlib import Path

from qwen_geo.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_reads_prompt_and_label_style(self) -> None:
        config = load_config(Path("configs/geoguessr_baseline.toml"))
        self.assertEqual(config.prompt_style, "strict_json")
        self.assertEqual(config.label_style, "full_json")
        self.assertIn(576, config.search.image_sizes)


if __name__ == "__main__":
    unittest.main()
