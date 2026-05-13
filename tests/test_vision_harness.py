from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import harness


class VisionHarnessTests(unittest.TestCase):
    def tearDown(self) -> None:
        harness.set_token_provider(None)

    def test_parse_response_checks_not_slide_before_slide(self) -> None:
        parsed = harness.parse_response(
            "DESCRIPTION: A talking head fills the frame.\n"
            "EVIDENCE: TEXT: none | PERSON: yes | BLANK: no\n"
            "VERDICT: NOT SLIDE"
        )

        self.assertEqual(parsed["verdict"], "NOT SLIDE")
        self.assertEqual(parsed["description"], "A talking head fills the frame.")
        self.assertIn("PERSON: yes", parsed["evidence"])
        self.assertIsNone(parsed["title"])
        self.assertIsNone(parsed["build_stage"])
        self.assertIsNone(parsed["layout"])

    def test_parse_response_extracts_v3_slide_attributes(self) -> None:
        parsed = harness.parse_response(
            "DESCRIPTION: A slide fills the frame.\n"
            "VERDICT: SLIDE\n"
            "TITLE: Questions for Business Leaders\n"
            "BUILD_STAGE: full\n"
            "LAYOUT: full-screen"
        )

        self.assertEqual(parsed["verdict"], "SLIDE")
        self.assertEqual(parsed["title"], "Questions for Business Leaders")
        self.assertEqual(parsed["build_stage"], "full")
        self.assertEqual(parsed["layout"], "full-screen")

    def test_parse_response_missing_v3_slide_attributes_are_null(self) -> None:
        parsed = harness.parse_response(
            "description: A slide fills the frame.\n"
            "verdict: SLIDE\n"
            "title: Market Map\n"
        )

        self.assertEqual(parsed["verdict"], "SLIDE")
        self.assertEqual(parsed["description"], "A slide fills the frame.")
        self.assertEqual(parsed["title"], "Market Map")
        self.assertIsNone(parsed["build_stage"])
        self.assertIsNone(parsed["layout"])

    def test_parse_response_accepts_case_variants_for_slide_attributes(self) -> None:
        parsed = harness.parse_response(
            "DESCRIPTION: A slide with a diagram.\n"
            "VERDICT: SLIDE\n"
            "title: Decision Tree\n"
            "Build_Stage: transitioning\n"
            "layout: split-right\n"
        )

        self.assertEqual(parsed["title"], "Decision Tree")
        self.assertEqual(parsed["build_stage"], "transitioning")
        self.assertEqual(parsed["layout"], "split-right")

    def test_successful_image_result_stores_slide_attributes_as_json_fields(self) -> None:
        result = harness.successful_image_result(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "DESCRIPTION: A slide fills the frame.\n"
                                "VERDICT: SLIDE\n"
                                "TITLE: Questions for Business Leaders\n"
                                "BUILD_STAGE: full\n"
                                "LAYOUT: full-screen"
                            )
                        }
                    }
                ],
                "usage": {"completion_tokens": 20, "prompt_tokens": 100},
            },
            response_time_sec=2.0,
        )

        self.assertEqual(result["title"], "Questions for Business Leaders")
        self.assertEqual(result["build_stage"], "full")
        self.assertEqual(result["layout"], "full-screen")

    def test_successful_image_result_stores_not_slide_attributes_as_null(self) -> None:
        result = harness.successful_image_result(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "DESCRIPTION: A person speaks to camera.\n"
                                "VERDICT: NOT SLIDE"
                            )
                        }
                    }
                ],
                "usage": {"completion_tokens": 10, "prompt_tokens": 100},
            },
            response_time_sec=2.0,
        )

        self.assertEqual(result["verdict"], "NOT SLIDE")
        self.assertIsNone(result["title"])
        self.assertIsNone(result["build_stage"])
        self.assertIsNone(result["layout"])

    def test_normalize_image_attribute_fields_backfills_older_runs(self) -> None:
        results = {
            "runs": [
                {
                    "images": [
                        {
                            "verdict": "SLIDE",
                            "title": " Market Map ",
                            "build_stage": "",
                        },
                        {
                            "verdict": "NOT SLIDE",
                            "title": "",
                            "build_stage": "",
                            "layout": "",
                        },
                    ]
                }
            ]
        }

        harness.normalize_image_attribute_fields(results)

        slide, not_slide = results["runs"][0]["images"]
        self.assertEqual(slide["title"], "Market Map")
        self.assertIsNone(slide["build_stage"])
        self.assertIsNone(slide["layout"])
        self.assertIsNone(not_slide["title"])
        self.assertIsNone(not_slide["build_stage"])
        self.assertIsNone(not_slide["layout"])

    def test_discover_images_sorts_by_filename_and_skips_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ["b.png", "notes.txt", "a.jpg", "c.jpeg"]:
                (root / name).write_bytes(b"x")

            self.assertEqual(
                [path.name for path in harness.discover_images(root)],
                ["a.jpg", "b.png", "c.jpeg"],
            )

    def test_batch_stats_counts_agreement_and_errors(self) -> None:
        stats = harness.calculate_batch_stats(
            [
                {"verdict": "SLIDE", "tokens_per_sec": 10, "response_time_sec": 2, "agreement": True},
                {"verdict": "NOT SLIDE", "tokens_per_sec": 20, "response_time_sec": 1, "agreement": False},
                {"verdict": "ERROR", "tokens_per_sec": 0, "response_time_sec": 5, "agreement": None},
            ],
            total_time_sec=8,
        )

        self.assertEqual(stats["slide_count"], 1)
        self.assertEqual(stats["not_slide_count"], 1)
        self.assertEqual(stats["error_count"], 1)
        self.assertEqual(stats["codex_agreement_count"], 1)
        self.assertEqual(stats["codex_disagreement_count"], 1)
        self.assertEqual(stats["codex_agreement_rate"], 0.5)

    def test_report_contains_all_required_sections(self) -> None:
        results = {
            "batch_path": "/tmp/images",
            "created_at": "2026-05-04T14:00:00Z",
            "codex_model": "baseline",
            "codex_verdicts": {"slide_0001.jpg": "SLIDE"},
            "runs": [
                {
                    "run_id": "run_001",
                    "model": "vision-model",
                    "prompt": "classify_v1",
                    "prompt_hash": "abc123",
                    "run_timestamp": "2026-05-04T14:32:00Z",
                    "batch_stats": {
                        "total_images": 1,
                        "total_time_sec": 1.5,
                        "avg_tokens_per_sec": 12.0,
                        "min_tokens_per_sec": 12.0,
                        "max_tokens_per_sec": 12.0,
                        "avg_response_time_sec": 1.5,
                        "slide_count": 0,
                        "not_slide_count": 1,
                        "error_count": 0,
                        "uncertain_count": 0,
                        "codex_agreement_count": 0,
                        "codex_disagreement_count": 1,
                        "codex_agreement_rate": 0.0,
                    },
                    "images": [
                        {
                            "filename": "slide_0001.jpg",
                            "description": "A blank title card.",
                            "evidence": "TEXT: none | PERSON: no | BLANK: yes",
                            "verdict": "NOT SLIDE",
                            "title": None,
                            "build_stage": None,
                            "layout": None,
                            "codex_verdict": "SLIDE",
                            "agreement": False,
                            "response_time_sec": 1.5,
                            "tokens_per_sec": 12.0,
                            "completion_tokens": 18,
                            "prompt_tokens": 100,
                            "raw_response": "VERDICT: NOT SLIDE",
                        }
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.md"
            report = harness.generate_report(results, report_path)

        self.assertIn("## Performance Matrix", report)
        self.assertIn("## Speed vs Accuracy", report)
        self.assertIn("## Per-Image Results", report)
        self.assertIn("## Disagreement Detail", report)
        self.assertIn("## Per-Run Detail", report)
        self.assertIn("NOT SLIDE ❌", report)
        self.assertIn("| Image | Verdict | Title | Build Stage | Layout | Codex | Agree | Time (s) | tok/s |", report)
        self.assertIn("| slide_0001.jpg | NOT SLIDE | - | - | - | SLIDE | no | 1.5 | 12.0 |", report)

    def test_request_headers_prefers_env_token(self) -> None:
        harness.set_token_provider(lambda: "keychain-token")

        with patch.dict("os.environ", {"LM_STUDIO_API_KEY": "env-token"}, clear=False):
            headers = harness.request_headers("application/json")

        self.assertEqual(headers["Authorization"], "Bearer env-token")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_request_headers_uses_token_provider(self) -> None:
        harness.set_token_provider(lambda: "keychain-token")

        with patch.dict("os.environ", {"LM_STUDIO_API_KEY": "", "LM_API_TOKEN": ""}, clear=False):
            headers = harness.request_headers()

        self.assertEqual(headers["Authorization"], "Bearer keychain-token")

    def test_model_level_failure_detection(self) -> None:
        result = harness.error_image_result(
            "LM Studio returned HTTP 400: No models loaded. Please load a model.",
            response_time_sec=0.1,
        )

        self.assertTrue(harness.is_model_level_failure(result))

    def test_known_codex_verdicts_ignores_empty_placeholders(self) -> None:
        results = {
            "codex_verdicts": {
                "a.png": "SLIDE",
                "b.png": "",
                "c.png": "UNCERTAIN",
            }
        }

        self.assertEqual(harness.known_codex_verdicts(results), {"a.png": "SLIDE"})

    def test_error_counts_as_disagreement_when_baseline_exists(self) -> None:
        result = harness.error_image_result("Model failed.", response_time_sec=0.1)
        result["filename"] = "slide.png"

        harness.attach_codex_agreement(result, {"slide.png": "SLIDE"})

        self.assertFalse(result["agreement"])

    def test_profile_builders_apply_hq_and_turbo_parameters(self) -> None:
        messages = [{"role": "user", "content": [{"type": "text", "text": "prompt"}]}]

        hq_load = harness.build_load_config("VISION_HQ", "vision-model")
        turbo_payload = harness.build_inference_payload("VISION_TURBO", "vision-model", messages)

        self.assertEqual(hq_load["model"], "vision-model")
        self.assertEqual(hq_load["context_length"], 32768)
        self.assertFalse(hq_load["offload_kv_cache_to_gpu"])
        self.assertEqual(turbo_payload["model"], "vision-model")
        self.assertEqual(turbo_payload["max_tokens"], 175)
        self.assertEqual(turbo_payload["top_k"], 1)

    def test_turbo_load_profile_uses_m3_ultra_values(self) -> None:
        turbo_load = harness.build_load_config("VISION_TURBO", "vision-model")

        self.assertEqual(turbo_load["context_length"], 16384)
        self.assertEqual(turbo_load["eval_batch_size"], 1024)
        self.assertFalse(turbo_load["offload_kv_cache_to_gpu"])

    def test_summarization_profiles_use_non_greedy_settings(self) -> None:
        summary_load = harness.build_load_config("SUMMARIZATION_HQ", "text-model")
        summary_payload = harness.build_inference_payload(
            "SUMMARIZATION_HQ",
            "text-model",
            [{"role": "user", "content": "Summarize this."}],
            max_tokens_override=256,
            stop=["END"],
        )

        self.assertEqual(summary_load["context_length"], 65536)
        self.assertFalse(summary_load["offload_kv_cache_to_gpu"])
        self.assertEqual(summary_payload["temperature"], 0.3)
        self.assertEqual(summary_payload["top_k"], 40)
        self.assertEqual(summary_payload["repeat_penalty"], 1.15)
        self.assertEqual(summary_payload["max_tokens"], 256)
        self.assertEqual(summary_payload["stop"], ["END"])

    def test_summarization_turbo_profile_uses_shorter_context_and_output(self) -> None:
        summary_load = harness.build_load_config("SUMMARIZATION_TURBO", "text-model")
        summary_payload = harness.build_inference_payload(
            "SUMMARIZATION_TURBO",
            "text-model",
            [{"role": "user", "content": "Summarize this."}],
        )

        self.assertEqual(summary_load["context_length"], 32768)
        self.assertEqual(summary_load["eval_batch_size"], 1024)
        self.assertEqual(summary_payload["max_tokens"], 512)
        self.assertNotIn("stop", summary_payload)

    def test_applied_load_config_mismatch_is_error(self) -> None:
        with self.assertRaises(harness.HarnessError):
            harness.validate_applied_load_config("VISION_HQ", {"context_length": 8192})

    def test_unknown_profile_is_user_facing_error(self) -> None:
        with self.assertRaises(harness.HarnessError):
            harness.normalize_profile_name("FASTISH")


if __name__ == "__main__":
    unittest.main()
