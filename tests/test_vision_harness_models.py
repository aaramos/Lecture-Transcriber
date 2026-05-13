from __future__ import annotations

import unittest

import harness


class VisionHarnessModelTests(unittest.TestCase):
    def test_extract_model_infos_detects_vision_metadata_and_loaded_state(self) -> None:
        payload = {
            "models": [
                {
                    "key": "qwen2.5-vl-32b-instruct",
                    "type": "llm",
                    "capabilities": ["text", "vision"],
                    "loaded_instances": [{"id": "qwen2.5-vl-32b-instruct:1"}],
                },
                {
                    "key": "text-only",
                    "type": "llm",
                    "capabilities": ["text"],
                    "loaded_instances": [],
                },
                {
                    "key": "embedding-model",
                    "type": "embedding",
                },
            ]
        }

        models = harness.extract_model_infos(native_payload=payload, openai_payload={})

        self.assertEqual([model["id"] for model in models], ["qwen2.5-vl-32b-instruct", "text-only"])
        self.assertTrue(models[0]["vision"])
        self.assertTrue(models[0]["loaded"])
        self.assertFalse(models[1]["vision"])

    def test_model_name_heuristic_marks_common_vision_models(self) -> None:
        vision, reason = harness.model_is_vision_capable("llava-v1.6-mistral", {})

        self.assertTrue(vision)
        self.assertIn("model name", reason)

    def test_loaded_model_rows_extracts_instances(self) -> None:
        rows = harness.loaded_model_rows(
            {
                "models": [
                    {"key": "model-a", "loaded_instances": [{"id": "model-a:1"}]},
                    {"key": "model-b", "loaded_instances": []},
                ]
            }
        )

        self.assertEqual(rows, [{"model": "model-a", "instance_id": "model-a:1"}])


if __name__ == "__main__":
    unittest.main()
