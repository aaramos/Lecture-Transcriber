import json
import unittest
from pathlib import Path


class TauriBundleConfigTests(unittest.TestCase):
    def test_bundle_uses_clean_staged_resources_and_local_signature(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "src-tauri" / "tauri.conf.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        resources = config["bundle"]["resources"]
        preparation_script = (root / "scripts" / "prepare-bundle-resources.mjs").read_text(
            encoding="utf-8"
        )

        self.assertEqual(1, len(resources))
        self.assertTrue("bundle-resources/project/" in resources)
        self.assertEqual("project/", resources["bundle-resources/project/"])
        self.assertEqual("-", config["bundle"]["macOS"]["signingIdentity"])
        self.assertIn('"git", ["ls-files", "src/lecture_processor/**"]', preparation_script)
        self.assertIn('".tools/darwin_arm64/ffmpeg"', preparation_script)
        self.assertIn('".tools/darwin_arm64/ffprobe"', preparation_script)


if __name__ == "__main__":
    unittest.main()
