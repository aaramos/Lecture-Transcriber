import json
import unittest
from pathlib import Path


class TauriBundleConfigTests(unittest.TestCase):
    def test_bundles_media_tools_for_packaged_app(self):
        config_path = Path(__file__).resolve().parents[1] / "src-tauri" / "tauri.conf.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        resources = set(config["bundle"]["resources"])

        self.assertIn("../.tools/darwin_arm64/ffmpeg", resources)
        self.assertIn("../.tools/darwin_arm64/ffprobe", resources)


if __name__ == "__main__":
    unittest.main()
