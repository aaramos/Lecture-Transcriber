import json
import tempfile
import unittest
import urllib.error
from unittest import mock
from pathlib import Path

from lecture_processor.ai.providers.base import (
    AnalyzeLectureRequest,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTransientError,
)
from lecture_processor.ai.providers.mlx_openai import (
    MLXTextProvider,
    MLXVisionProvider,
    _OpenAICompatibleClient,
    _Usage,
    _lm_studio_native_base_url,
    _lm_studio_openai_base_url,
    _loaded_model_instances_from_payload,
    _model_ids_from_payload,
    _resources_from_gathered_context,
    _structured_json_from_message,
)


class MLXOpenAIProviderTests(unittest.TestCase):
    def setUp(self):
        _OpenAICompatibleClient._loaded_instance_cache = {}

    def test_text_provider_parses_openai_compatible_overview_response(self):
        provider = MLXTextProvider(base_url="http://local.test/v1", model="default")
        provider._client = FakeClient(
            {
                "title": "Local Model Lecture",
                "executive_summary": "A local summary.",
                "outline": [{"id": 1, "heading": "Intro", "slide_ids": [1]}],
                "warnings": [],
            }
        )

        overview = provider.analyze_overview(_request())

        self.assertEqual(overview["title"], "Local Model Lecture")
        self.assertEqual(overview["outline"][0]["heading"], "Intro")
        self.assertGreater(overview["input_tokens"], 0)
        self.assertIn("local LM Studio text", " ".join(overview["warnings"]))

    def test_overview_prompt_trims_transcript_to_16000_chars(self):
        provider = MLXTextProvider(base_url="http://local.test/v1", model="default")
        provider._client = FakeClient(
            {
                "title": "Trimmed Lecture",
                "executive_summary": "A summary.",
                "outline": [{"id": 1, "heading": "Intro", "slide_ids": [1]}],
                "warnings": [],
            }
        )

        provider.analyze_overview(_request(transcript_text="word " * 5000))

        prompt = provider._client.messages[1]["content"]
        self.assertIn("[trimmed]", prompt)
        self.assertLess(len(prompt), 17000)

    def test_transcript_prompt_requests_readable_paragraphs(self):
        provider = MLXTextProvider(base_url="http://local.test/v1", model="default")
        provider._client = FakeClient({"formatted_transcript": "First paragraph.\n\nSecond paragraph.", "warnings": []})

        transcript = provider.analyze_transcript(_request())

        self.assertIn("Second paragraph", transcript["formatted_transcript"])
        prompt = provider._client.messages[1]["content"]
        self.assertIn("Add paragraph breaks every 2-5 sentences", prompt)
        self.assertIn("not one large blob", prompt)

    def test_vision_provider_sends_slide_image_as_native_data_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slide_path = root / "slide.png"
            slide_path.write_bytes(b"not-a-real-png-but-valid-test-bytes")
            provider = MLXVisionProvider(base_url="http://local.test/v1", model="default")
            provider._client = FakeClient(
                {
                    "slide_analysis": [
                        {
                            "slide_id": 1,
                            "descriptive_filename": "slide_0001.png",
                            "caption": "A title slide.",
                            "summary": "The slide introduces the lecture.",
                            "tags": ["intro"],
                            "instructor_commentary": "The instructor opens the lecture.",
                        }
                    ],
                    "warnings": [],
                }
            )

            result = provider.analyze_slides(_request(lecture_dir=root))

            self.assertEqual(result["slide_analysis"][0]["caption"], "A title slide.")
            content = provider._client.image_input
            image_items = [item for item in content if item.get("type") == "image"]
            self.assertEqual(len(image_items), 1)
            self.assertTrue(image_items[0]["data_url"].startswith("data:image/png;base64,"))

    def test_vision_provider_uses_optimized_ai_image_without_replacing_source(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slide_path = root / "slide.png"
            Image.new("RGB", (1600, 900), color=(255, 255, 255)).save(slide_path)
            original_size = slide_path.stat().st_size
            provider = MLXVisionProvider(base_url="http://local.test/v1", model="default")
            provider._client = FakeClient(
                {
                    "slide_analysis": [
                        {
                            "slide_id": 1,
                            "descriptive_filename": "slide_0001.png",
                            "caption": "A title slide.",
                            "summary": "The slide introduces the lecture.",
                            "tags": ["intro"],
                            "instructor_commentary": "The instructor opens the lecture.",
                        }
                    ],
                    "warnings": [],
                }
            )

            provider.analyze_slides(_request(lecture_dir=root))

            content = provider._client.image_input
            image_items = [item for item in content if item.get("type") == "image"]
            self.assertEqual(len(image_items), 1)
            self.assertTrue(image_items[0]["data_url"].startswith("data:image/webp;base64,"))
            cached_slide = root / ".ai-cache" / "slide.webp"
            self.assertTrue(cached_slide.exists())
            with Image.open(cached_slide) as ai_image:
                self.assertLessEqual(max(ai_image.size), 1024)
            with Image.open(slide_path) as source_image:
                self.assertEqual((1600, 900), source_image.size)
            self.assertEqual(original_size, slide_path.stat().st_size)

    def test_vision_provider_retries_missing_slide_ids_once(self):
        request = AnalyzeLectureRequest(
            lecture_id="lecture",
            transcript_text="slide one and slide two",
            segments=[
                {"id": 1, "text": "slide one"},
                {"id": 2, "text": "slide two"},
            ],
            slides=[
                {"id": 1, "linked_segment_ids": [1]},
                {"id": 2, "linked_segment_ids": [2]},
            ],
            duration_minutes=2.0,
        )
        provider = MLXVisionProvider(base_url="http://local.test/v1", model="default")
        provider._client = SequenceImageClient(
            [
                {
                    "slide_analysis": [
                        {
                            "slide_id": 1,
                            "descriptive_filename": "slide_0001.png",
                            "caption": "Slide one.",
                            "summary": "First slide.",
                            "tags": ["one"],
                            "instructor_commentary": "First notes.",
                        }
                    ],
                    "warnings": [],
                },
                {
                    "slide_analysis": [
                        {
                            "slide_id": 2,
                            "descriptive_filename": "slide_0002.png",
                            "caption": "Slide two.",
                            "summary": "Second slide.",
                            "tags": ["two"],
                            "instructor_commentary": "Second notes.",
                        }
                    ],
                    "warnings": [],
                },
            ]
        )

        result = provider.analyze_slides(request)

        self.assertEqual(["Slide one.", "Slide two."], [item["caption"] for item in result["slide_analysis"]])
        self.assertEqual(2, len(provider._client.image_inputs))
        self.assertEqual("text", provider._client.image_inputs[1][0]["type"])
        self.assertIn("slide_id=2", provider._client.image_inputs[1][0]["content"])
        self.assertNotIn("slide_id=1", provider._client.image_inputs[1][0]["content"])

    def test_image_chat_uses_lm_studio_native_api_shape(self):
        captured = {}

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured.setdefault("urls", []).append(request.full_url)
            if request.full_url.endswith("/models"):
                return FakeResponse({"models": [{"type": "llm", "key": "gemma", "loaded_instances": [{"id": "gemma-live"}]}]})
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "output": [{"type": "message", "content": "{\"ok\": true}"}],
                    "stats": {"input_tokens": 3, "total_output_tokens": 2},
                }
            )

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            payload, usage = client.chat_json_with_lm_studio_images(
                [
                    {"type": "text", "content": "Describe slide 1."},
                    {"type": "image", "data_url": "data:image/png;base64,aW1hZ2U="},
                ],
                system="Return JSON.",
                max_tokens=32,
                temperature=0.0,
                context="test vision",
            )

        self.assertEqual({"ok": True}, payload)
        self.assertEqual(3, usage.input_tokens)
        self.assertEqual("http://localhost:1234/api/v1/chat", captured["url"])
        self.assertEqual(
            [
                "http://localhost:1234/api/v1/models",
                "http://localhost:1234/api/v1/chat",
            ],
            captured["urls"],
        )
        self.assertEqual("Return JSON.", captured["body"]["system_prompt"])
        self.assertEqual({"type": "text", "content": "Describe slide 1."}, captured["body"]["input"][0])
        self.assertEqual("data:image/png;base64,aW1hZ2U=", captured["body"]["input"][1]["data_url"])
        self.assertNotIn("context_length", captured["body"])
        self.assertNotIn("reasoning", captured["body"])
        self.assertNotIn("messages", captured["body"])

    def test_ollama_requests_disable_thinking(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [{"message": {"content": "{\"ok\": true}"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        client = _OpenAICompatibleClient("http://localhost:11434/v1", "gemma4:26b", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            text, usage = client.chat_text(
                [{"role": "user", "content": "Return JSON."}],
                max_tokens=32,
                temperature=0.0,
                context="test",
            )

        self.assertEqual("{\"ok\": true}", text)
        self.assertEqual(1, usage.input_tokens)
        self.assertIs(captured["body"]["think"], False)
        self.assertIs(captured["body"]["stream"], False)

    def test_default_model_uses_first_available_lm_studio_model(self):
        captured = {}

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured.setdefault("urls", []).append(request.full_url)
            if request.full_url.endswith("/models"):
                return FakeResponse(
                    {
                        "models": [
                            {"type": "llm", "key": "qwen3-8b", "loaded_instances": [{"id": "qwen3-8b"}]},
                            {"type": "llm", "key": "gemma-3-12b-it", "loaded_instances": []},
                        ]
                    }
                )
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "output": [{"type": "message", "content": "{\"ok\": true}"}],
                    "stats": {"input_tokens": 1, "total_output_tokens": 1},
                }
            )

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "default", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            text, _usage = client.chat_text(
                [{"role": "user", "content": "Return JSON."}],
                max_tokens=32,
                temperature=0.0,
                context="test",
            )

        self.assertEqual("{\"ok\": true}", text)
        self.assertEqual("qwen3-8b", captured["body"]["model"])
        self.assertNotIn("context_length", captured["body"])
        self.assertNotIn("reasoning", captured["body"])
        self.assertEqual(
            [
                "http://localhost:1234/api/v1/models",
                "http://localhost:1234/api/v1/chat",
            ],
            captured["urls"],
        )

    def test_lm_studio_api_token_uses_authorization_header(self):
        captured = {}

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            if request.full_url.endswith("/models"):
                return FakeResponse({"models": [{"type": "llm", "key": "gemma", "loaded_instances": [{"id": "gemma-live"}]}]})
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({"output": [{"type": "message", "content": "{\"ok\": true}"}], "stats": {}})

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch.dict("os.environ", {"LM_STUDIO_API_KEY": "secret-token"}, clear=False):
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                client.chat_text(
                    [{"role": "user", "content": "Return JSON."}],
                    max_tokens=32,
                    temperature=0.0,
                    context="test",
                )

        self.assertEqual("Bearer secret-token", captured["headers"]["Authorization"])
        self.assertEqual("gemma-live", captured["body"]["model"])

    def test_tool_chat_uses_lm_studio_native_api_and_integrations(self):
        captured = {}

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            if request.full_url.endswith("/models"):
                return FakeResponse({"models": [{"type": "llm", "key": "gemma", "loaded_instances": [{"id": "gemma-live"}]}]})
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "output": [
                        {"type": "tool_call", "tool": "brave_web_search", "arguments": {}, "output": "{}"},
                        {"type": "message", "content": "{\"resources\": [], \"warnings\": []}"},
                    ],
                    "stats": {"input_tokens": 12, "total_output_tokens": 8},
                }
            )

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            payload, usage, tools = client.chat_json_with_lm_studio_tools(
                "Find resources.",
                system="Search first.",
                max_tokens=256,
                temperature=0.0,
                context="test resources",
            )

        self.assertEqual({"resources": [], "warnings": []}, payload)
        self.assertEqual(12, usage.input_tokens)
        self.assertEqual(8, usage.output_tokens)
        self.assertEqual(["brave_web_search"], tools)
        self.assertEqual("http://localhost:1234/api/v1/chat", captured["url"])
        self.assertEqual("gemma-live", captured["body"]["model"])
        self.assertEqual("Search first.", captured["body"]["system_prompt"])
        self.assertEqual("Find resources.", captured["body"]["input"])
        self.assertNotIn("reasoning", captured["body"])
        self.assertNotIn("Authorization", captured["headers"])
        integration_ids = [item["id"] for item in captured["body"]["integrations"]]
        self.assertEqual(["mcp/brave-search", "mcp/fetch"], integration_ids)

    def test_tool_chat_can_use_tool_output_when_model_returns_no_message(self):
        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            if request.full_url.endswith("/models"):
                return FakeResponse({"models": [{"type": "llm", "key": "gemma", "loaded_instances": [{"id": "gemma"}]}]})
            return FakeResponse(
                {
                    "output": [
                        {
                            "type": "tool_call",
                            "tool": "brave_web_search",
                            "arguments": {},
                            "output": "Title: Resource\nDescription: Useful.\nURL: https://example.com",
                        },
                        {"type": "message", "content": "\n\n"},
                    ],
                    "stats": {"input_tokens": 12, "total_output_tokens": 8},
                }
            )

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            text, _usage, tools = client.chat_text_with_lm_studio_tools(
                "Find resources.",
                system="Search first.",
                max_tokens=256,
                temperature=0.0,
                context="test resources",
            )

        self.assertIn("Title: Resource", text)
        self.assertEqual(["brave_web_search"], tools)

    def test_native_chat_does_not_send_reasoning_for_lm_studio_models(self):
        captured = {"chat_bodies": []}

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            if request.full_url.endswith("/models"):
                return FakeResponse({"models": [{"type": "llm", "key": "gemma", "loaded_instances": [{"id": "gemma"}]}]})
            body = json.loads(request.data.decode("utf-8"))
            captured["chat_bodies"].append(body)
            return FakeResponse({"output": [{"type": "message", "content": "ok"}], "stats": {}})

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            text, _usage, _tools = client.chat_text_with_lm_studio_native(
                "Hello.",
                system="System.",
                max_tokens=32,
                temperature=0.0,
                context="test",
            )

        self.assertEqual("ok", text)
        self.assertEqual(1, len(captured["chat_bodies"]))
        self.assertNotIn("reasoning", captured["chat_bodies"][0])

    def test_structured_json_parser_reads_message_content(self):
        parsed = _structured_json_from_message(
            {"content": '{"ok": true}', "reasoning_content": ""},
            context="test structured",
        )

        self.assertEqual({"ok": True}, parsed)

    def test_structured_json_parser_reads_reasoning_when_content_empty(self):
        parsed = _structured_json_from_message(
            {"content": "", "reasoning_content": '{"ok": true}'},
            context="test structured",
        )

        self.assertEqual({"ok": True}, parsed)

    def test_structured_json_parser_uses_reasoning_when_content_is_invalid(self):
        parsed = _structured_json_from_message(
            {"content": "not json", "reasoning_content": '{"ok": true}'},
            context="test structured",
        )

        self.assertEqual({"ok": True}, parsed)

    def test_structured_chat_posts_json_schema_response_format(self):
        captured = {"bodies": []}

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            if request.full_url.endswith("/models"):
                return FakeResponse({"models": [{"type": "llm", "key": "gemma", "loaded_instances": [{"id": "gemma"}]}]})
            captured["url"] = request.full_url
            body = json.loads(request.data.decode("utf-8"))
            captured["bodies"].append(body)
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning_content": '{"queries":[{"query":"ai creativity","intent":"topic"}],"warnings":[]}',
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 7},
                }
            )

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            payload, usage = client.chat_structured_json(
                [{"role": "user", "content": "Plan"}],
                schema_name="resource_query_plan",
                schema={"type": "object"},
                max_tokens=128,
                temperature=0.1,
                context="test structured",
            )

        self.assertEqual("http://localhost:1234/v1/chat/completions", captured["url"])
        self.assertEqual("json_schema", captured["bodies"][0]["response_format"]["type"])
        self.assertTrue(captured["bodies"][0]["response_format"]["json_schema"]["strict"])
        self.assertEqual("ai creativity", payload["queries"][0]["query"])
        self.assertEqual(10, usage.input_tokens)

    def test_tool_chat_explains_lm_studio_plugin_permission_error(self):
        class FakeErrorBody:
            def read(self):
                return b'{"error":{"message":"Permission denied to use plugin \\"mcp/brave-search\\"."}}'

            def close(self):
                return None

        def fake_urlopen(request, timeout):
            if request.full_url.endswith("/models"):
                class FakeResponse:
                    def __enter__(self):
                        return self

                    def __exit__(self, *_args):
                        return False

                    def read(self):
                        return json.dumps(
                            {"models": [{"type": "llm", "key": "gemma", "loaded_instances": [{"id": "gemma"}]}]}
                        ).encode("utf-8")

                return FakeResponse()
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, FakeErrorBody())

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(ProviderRequestError) as raised:
                client.chat_json_with_lm_studio_tools(
                    "Find resources.",
                    system="Search first.",
                    max_tokens=256,
                    temperature=0.0,
                    context="test resources",
                )

        self.assertIn("Allow calling servers from mcp.json", str(raised.exception))

    def test_resource_prompt_uses_direct_search_then_lm_studio_formatting(self):
        provider = MLXTextProvider(base_url="http://local.test/v1", model="default")
        provider._client = FakeClient(
            {
                "resources": [
                    {
                        "title": "Official docs",
                        "url": "https://example.com/docs",
                        "summary": "A verified resource.",
                        "source_quality": "high",
                    }
                ],
                "warnings": [],
            },
            tool_calls=["brave_web_search", "fetch_url_content_tool"],
        )

        direct_search = mock.Mock(
            return_value=(
                [
                    {
                        "title": "Official docs",
                        "url": "https://example.com/docs",
                        "summary": "A verified resource.",
                        "source_quality": "high",
                    }
                ],
                [],
            )
        )
        with mock.patch(
            "lecture_processor.ai.providers.mlx_openai._direct_resource_candidates",
            direct_search,
        ):
            resources = provider.analyze_resources(_request())

        self.assertEqual("Official docs", resources["resources"][0]["title"])
        self.assertIn("direct Brave Search", " ".join(resources["warnings"]))
        self.assertIn("Resource queries planned", " ".join(resources["warnings"]))
        self.assertNotIn("not web-grounded", " ".join(resources["warnings"]))
        self.assertIsNone(provider._client.tool_prompt)
        self.assertIn("Verified resource candidates", provider._client.format_prompt)
        self.assertNotIn("Transcript excerpt", provider._client.format_prompt)
        self.assertEqual(["official lecture query"], direct_search.call_args.kwargs["queries"])

    def test_resource_query_planner_caps_queries_to_six(self):
        provider = MLXTextProvider(base_url="http://local.test/v1", model="default")
        provider._client = FakeClient({"resources": [], "warnings": []})
        provider._client.planner_payload = {
            "queries": [
                {"query": f"query {index}", "intent": "topic"}
                for index in range(10)
            ],
            "warnings": [],
        }

        direct_search = mock.Mock(return_value=([], ["No direct resources."]))
        with mock.patch("lecture_processor.ai.providers.mlx_openai._direct_resource_candidates", direct_search):
            provider.analyze_resources(_request())

        self.assertEqual(
            [f"query {index}" for index in range(6)],
            direct_search.call_args.kwargs["queries"],
        )

    def test_resource_query_planner_failure_uses_heuristic_queries(self):
        provider = MLXTextProvider(base_url="http://local.test/v1", model="default")
        provider._client = FailingPlannerClient()

        direct_search = mock.Mock(return_value=([], ["No direct resources."]))
        with mock.patch("lecture_processor.ai.providers.mlx_openai._direct_resource_candidates", direct_search):
            resources = provider.analyze_resources(_request())

        self.assertEqual([], direct_search.call_args.kwargs["queries"])
        self.assertIn("Resource query planner failed", " ".join(resources["warnings"]))

    def test_resources_fallback_extracts_verified_tool_results_when_json_format_fails(self):
        provider = MLXTextProvider(base_url="http://local.test/v1", model="default")
        provider._client = FailingResourceFormatClient()

        with mock.patch(
            "lecture_processor.ai.providers.mlx_openai._direct_resource_candidates",
            return_value=([], ["Direct search unavailable."]),
        ):
            resources = provider.analyze_resources(_request())

        self.assertEqual("Human-AI Collaboration: What is it and Why is it Important? | IBM", resources["resources"][0]["title"])
        self.assertEqual("https://www.ibm.com/think/topics/human-ai-collaboration", resources["resources"][0]["url"])
        self.assertEqual("high", resources["resources"][0]["source_quality"])
        self.assertIn("used verified LM Studio web-tool results", " ".join(resources["warnings"]))

    def test_resources_extract_from_lm_studio_markdown_output(self):
        resources = _resources_from_gathered_context(
            """
## 1. **AI in Advertising: How It's Transforming Marketing in 2026**
**URL:** https://www.stackadapt.com/resources/blog/ai-advertising
**Relevance:** This resource explains how AI is used for personalized ad copy and audience targeting.
**Source Quality:** High

## 2. **LeewayHertz**
**Title:** AI in Product Lifecycle Management: Applications, Industry Use Cases
**URL:** https://www.leewayhertz.com/ai-in-product-lifecycle-management/
**Relevance:** This report maps AI applications across design and production.
**Source Quality:** Medium
"""
        )

        self.assertEqual(2, len(resources))
        self.assertEqual("AI in Advertising: How It's Transforming Marketing in 2026", resources[0]["title"])
        self.assertEqual("https://www.stackadapt.com/resources/blog/ai-advertising", resources[0]["url"])
        self.assertIn("personalized ad copy", resources[0]["summary"])

    def test_resources_extract_when_lm_studio_reports_links_without_tool_calls(self):
        provider = MLXTextProvider(base_url="http://local.test/v1", model="default")
        provider._client = FakeClient(
            {
                "resources": [
                    {
                        "title": "Untrusted docs",
                        "url": "https://example.com/docs",
                        "summary": "A resource.",
                        "source_quality": "medium",
                    }
                ],
                "warnings": [],
            },
            tool_calls=[],
        )

        with mock.patch(
            "lecture_processor.ai.providers.mlx_openai._direct_resource_candidates",
            return_value=([], ["Direct search unavailable."]),
        ):
            resources = provider.analyze_resources(_request())

        self.assertEqual("Official docs", resources["resources"][0]["title"])
        self.assertEqual("https://example.com/docs", resources["resources"][0]["url"])
        self.assertIn("did not report web-search tool-call metadata", " ".join(resources["warnings"]))
        self.assertIsNone(provider._client.format_prompt)

    def test_direct_resources_save_verified_candidates_when_formatter_fails(self):
        provider = MLXTextProvider(base_url="http://local.test/v1", model="default")
        provider._client = FailingResourceFormatClient()

        with mock.patch(
            "lecture_processor.ai.providers.mlx_openai._direct_resource_candidates",
            return_value=(
                [
                    {
                        "title": "Verified candidate",
                        "url": "https://example.com/verified",
                        "summary": "A verified search result.",
                        "source_quality": "medium",
                    }
                ],
                [],
            ),
        ):
            resources = provider.analyze_resources(_request())

        self.assertEqual("Verified candidate", resources["resources"][0]["title"])
        self.assertEqual("https://example.com/verified", resources["resources"][0]["url"])
        self.assertIn("saved verified search results", " ".join(resources["warnings"]))
        self.assertNotIn("Resources route is off", " ".join(resources["warnings"]))

    def test_resource_formatter_filters_urls_not_in_verified_candidates(self):
        provider = MLXTextProvider(base_url="http://local.test/v1", model="default")
        provider._client = FakeClient(
            {
                "resources": [
                    {
                        "title": "Invented resource",
                        "url": "https://example.com/invented",
                        "summary": "Not from search.",
                        "source_quality": "medium",
                    }
                ],
                "warnings": [],
            }
        )

        with mock.patch(
            "lecture_processor.ai.providers.mlx_openai._direct_resource_candidates",
            return_value=(
                [
                    {
                        "title": "Verified candidate",
                        "url": "https://example.com/verified",
                        "summary": "A verified search result.",
                        "source_quality": "medium",
                    }
                ],
                [],
            ),
        ):
            resources = provider.analyze_resources(_request())

        self.assertEqual("Verified candidate", resources["resources"][0]["title"])
        self.assertEqual("https://example.com/verified", resources["resources"][0]["url"])
        self.assertIn("filled remaining slots from verified search results", " ".join(resources["warnings"]))

    def test_resource_timeout_reports_failure_without_route_off_warning(self):
        provider = MLXTextProvider(base_url="http://local.test/v1", model="default")
        provider._client = TimeoutResourceClient()

        with mock.patch(
            "lecture_processor.ai.providers.mlx_openai._direct_resource_candidates",
            return_value=([], ["Direct search unavailable."]),
        ):
            resources = provider.analyze_resources(_request())

        self.assertEqual([], resources["resources"])
        warnings = " ".join(resources["warnings"])
        self.assertIn("timed out", warnings)
        self.assertIn("Direct search unavailable", warnings)
        self.assertNotIn("Resources route is off", warnings)

    def test_lm_studio_native_base_url_from_openai_url(self):
        self.assertEqual("http://localhost:1234/api/v1", _lm_studio_native_base_url("http://localhost:1234/v1"))
        self.assertEqual("http://localhost:1234/api/v1", _lm_studio_native_base_url("http://localhost:1234/api/v1"))
        self.assertEqual("http://localhost:1234/api/v1", _lm_studio_native_base_url("http://localhost:1234"))

    def test_lm_studio_openai_base_url_from_native_url(self):
        self.assertEqual("http://localhost:1234/v1", _lm_studio_openai_base_url("http://localhost:1234/v1"))
        self.assertEqual("http://localhost:1234/v1", _lm_studio_openai_base_url("http://localhost:1234/api/v1"))
        self.assertEqual("http://localhost:1234/v1", _lm_studio_openai_base_url("http://localhost:1234"))

    def test_model_ids_from_lm_studio_payload(self):
        self.assertEqual(
            ["qwen3-8b", "gemma-3-12b-it"],
            _model_ids_from_payload({"data": [{"id": "qwen3-8b"}, {"id": "gemma-3-12b-it"}]}),
        )
        self.assertEqual(
            ["gemma-4-31b-it-mlx"],
            _model_ids_from_payload({"models": [{"key": "gemma-4-31b-it-mlx", "display_name": "Gemma 4"}]}),
        )
        self.assertEqual(
            ["gemma-4-31b-it-mlx"],
            _model_ids_from_payload(
                {
                    "models": [
                        {"type": "embedding", "key": "text-embedding-nomic-embed-text-v1.5"},
                        {"type": "llm", "key": "gemma-4-31b-it-mlx"},
                    ]
                }
            ),
        )

    def test_loaded_model_instances_from_lm_studio_payload(self):
        self.assertEqual(
            {"gemma-live": "gemma-live", "gemma": "gemma-live"},
            _loaded_model_instances_from_payload(
                {
                    "models": [
                        {"type": "embedding", "key": "embed", "loaded_instances": [{"id": "embed-live"}]},
                        {"type": "llm", "key": "gemma", "loaded_instances": [{"id": "gemma-live"}]},
                    ]
                }
            ),
        )

    def test_lm_studio_uses_loaded_model_without_loading_again(self):
        captured = {"urls": []}

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["urls"].append(request.full_url)
            if request.full_url.endswith("/models"):
                return FakeResponse({"models": [{"type": "llm", "key": "gemma", "loaded_instances": [{"id": "gemma-live"}]}]})
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({"output": [{"type": "message", "content": "ok"}], "stats": {}})

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            text, _usage, _tools = client.chat_text_with_lm_studio_native(
                "Hello.",
                system="System.",
                max_tokens=32,
                temperature=0.0,
                context="test",
            )

        self.assertEqual("ok", text)
        self.assertEqual(
            [
                "http://localhost:1234/api/v1/models",
                "http://localhost:1234/api/v1/chat",
            ],
            captured["urls"],
        )
        self.assertEqual("gemma-live", captured["body"]["model"])

    def test_lm_studio_loads_missing_model_once_and_reuses_cache(self):
        captured = {"urls": [], "load_bodies": [], "chat_models": []}
        state = {"model_loaded": False}

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["urls"].append(request.full_url)
            if request.full_url.endswith("/models"):
                loaded_instances = [{"id": "gemma-live"}] if state["model_loaded"] else []
                return FakeResponse({"models": [{"type": "llm", "key": "gemma", "loaded_instances": loaded_instances}]})
            if request.full_url.endswith("/models/load"):
                captured["load_bodies"].append(json.loads(request.data.decode("utf-8")))
                state["model_loaded"] = True
                return FakeResponse({"type": "llm", "instance_id": "gemma-live", "status": "loaded"})
            body = json.loads(request.data.decode("utf-8"))
            captured["chat_models"].append(body["model"])
            return FakeResponse({"output": [{"type": "message", "content": "ok"}], "stats": {}})

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            for _index in range(2):
                client.chat_text_with_lm_studio_native(
                    "Hello.",
                    system="System.",
                    max_tokens=32,
                    temperature=0.0,
                    context="test",
                )

        self.assertEqual([{"model": "gemma"}], captured["load_bodies"])
        self.assertEqual(["gemma-live", "gemma-live"], captured["chat_models"])
        self.assertEqual(
            [
                "http://localhost:1234/api/v1/models",
                "http://localhost:1234/api/v1/models/load",
                "http://localhost:1234/api/v1/models",
                "http://localhost:1234/api/v1/chat",
                "http://localhost:1234/api/v1/models",
                "http://localhost:1234/api/v1/chat",
            ],
            captured["urls"],
        )

    def test_lm_studio_unloads_competing_model_even_when_requested_model_is_loaded(self):
        captured = {"urls": [], "unload_bodies": [], "chat_models": []}
        loaded_instances = {
            "gemma": [{"id": "gemma-live"}],
            "vision": [{"id": "vision-live"}],
        }

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["urls"].append(request.full_url)
            if request.full_url.endswith("/models"):
                return FakeResponse(
                    {
                        "models": [
                            {"type": "llm", "key": "gemma", "loaded_instances": loaded_instances["gemma"]},
                            {"type": "llm", "key": "vision", "loaded_instances": loaded_instances["vision"]},
                        ]
                    }
                )
            if request.full_url.endswith("/models/unload"):
                body = json.loads(request.data.decode("utf-8"))
                captured["unload_bodies"].append(body)
                for key, instances in list(loaded_instances.items()):
                    loaded_instances[key] = [
                        instance for instance in instances if instance.get("id") != body.get("instance_id")
                    ]
                return FakeResponse({"status": "unloaded"})
            body = json.loads(request.data.decode("utf-8"))
            captured["chat_models"].append(body["model"])
            return FakeResponse({"output": [{"type": "message", "content": "ok"}], "stats": {}})

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.chat_text_with_lm_studio_native(
                "Hello.",
                system="System.",
                max_tokens=32,
                temperature=0.0,
                context="test",
            )

        self.assertEqual([{"instance_id": "vision-live"}], captured["unload_bodies"])
        self.assertEqual(["gemma-live"], captured["chat_models"])

    def test_lm_studio_unloads_duplicate_requested_model_instances(self):
        captured = {"unload_bodies": [], "chat_models": []}
        loaded_instances = [{"id": "gemma-live"}, {"id": "gemma-live:2"}]

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            if request.full_url.endswith("/models"):
                return FakeResponse(
                    {
                        "models": [
                            {
                                "type": "llm",
                                "key": "gemma",
                                "loaded_instances": loaded_instances,
                            }
                        ]
                    }
                )
            if request.full_url.endswith("/models/unload"):
                body = json.loads(request.data.decode("utf-8"))
                captured["unload_bodies"].append(body)
                loaded_instances[:] = [
                    instance for instance in loaded_instances if instance.get("id") != body.get("instance_id")
                ]
                return FakeResponse({"status": "unloaded"})
            body = json.loads(request.data.decode("utf-8"))
            captured["chat_models"].append(body["model"])
            return FakeResponse({"output": [{"type": "message", "content": "ok"}], "stats": {}})

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.chat_text_with_lm_studio_native(
                "Hello.",
                system="System.",
                max_tokens=32,
                temperature=0.0,
                context="test",
            )

        self.assertEqual([{"instance_id": "gemma-live:2"}], captured["unload_bodies"])
        self.assertEqual(["gemma-live"], captured["chat_models"])

    def test_lm_studio_blocks_chat_when_loaded_instance_state_is_missing(self):
        captured = {"chat_called": False}

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            if request.full_url.endswith("/models"):
                return FakeResponse({"data": [{"id": "gemma"}]})
            captured["chat_called"] = True
            return FakeResponse({"output": [{"type": "message", "content": "ok"}], "stats": {}})

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(ProviderRequestError) as raised:
                client.chat_text_with_lm_studio_native(
                    "Hello.",
                    system="System.",
                    max_tokens=32,
                    temperature=0.0,
                    context="test",
                )

        self.assertFalse(captured["chat_called"])
        self.assertIn("did not report loaded model instances", str(raised.exception))

    def test_lm_studio_blocks_chat_when_duplicate_instances_remain_after_unload(self):
        captured = {"unload_bodies": [], "chat_called": False}

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout):
            if request.full_url.endswith("/models"):
                return FakeResponse(
                    {
                        "models": [
                            {
                                "type": "llm",
                                "key": "gemma",
                                "loaded_instances": [{"id": "gemma-live"}, {"id": "gemma-live:2"}],
                            }
                        ]
                    }
                )
            if request.full_url.endswith("/models/unload"):
                captured["unload_bodies"].append(json.loads(request.data.decode("utf-8")))
                return FakeResponse({"status": "unloaded"})
            captured["chat_called"] = True
            return FakeResponse({"output": [{"type": "message", "content": "ok"}], "stats": {}})

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(ProviderRequestError) as raised:
                client.chat_text_with_lm_studio_native(
                    "Hello.",
                    system="System.",
                    max_tokens=32,
                    temperature=0.0,
                    context="test",
                )

        self.assertEqual([{"instance_id": "gemma-live:2"}, {"instance_id": "gemma-live:2"}], captured["unload_bodies"])
        self.assertFalse(captured["chat_called"])
        self.assertIn("does not have exactly one loaded model", str(raised.exception))

    def test_lm_studio_unloads_loaded_model_instance(self):
        captured = {}
        native_base = _lm_studio_native_base_url("http://localhost:1234/v1")
        _OpenAICompatibleClient._loaded_instance_cache[native_base] = {
            "gemma": "gemma-live",
            "gemma-live": "gemma-live",
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"instance_id": "gemma-live"}).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        client = _OpenAICompatibleClient("http://localhost:1234/v1", "gemma", 30)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            unloaded = client.unload_model("gemma")

        self.assertTrue(unloaded)
        self.assertEqual("http://localhost:1234/api/v1/models/unload", captured["url"])
        self.assertEqual({"instance_id": "gemma-live"}, captured["body"])
        self.assertNotIn("gemma", _OpenAICompatibleClient._loaded_instance_cache[native_base])
        self.assertNotIn("gemma-live", _OpenAICompatibleClient._loaded_instance_cache[native_base])


class FakeClient:
    def __init__(self, payload, *, tool_calls=None):
        self.payload = payload
        self.tool_calls = tool_calls or []
        self.planner_payload = {
            "queries": [{"query": "official lecture query", "intent": "Find official resources."}],
            "warnings": [],
        }
        self.messages = None
        self.tool_prompt = None
        self.format_prompt = None
        self.image_input = None

    def chat_json(self, messages, *, max_tokens, temperature, context):
        self.messages = messages
        return self.payload, _Usage(input_tokens=12, output_tokens=max(1, len(json.dumps(self.payload)) // 4))

    def chat_json_with_lm_studio_tools(self, prompt, *, system, max_tokens, temperature, context):
        self.tool_prompt = prompt
        return self.payload, _Usage(input_tokens=12, output_tokens=max(1, len(json.dumps(self.payload)) // 4)), self.tool_calls

    def chat_text_with_lm_studio_tools(self, prompt, *, system, max_tokens, temperature, context):
        self.tool_prompt = prompt
        return "Official docs: https://example.com/docs is relevant and high quality.", _Usage(
            input_tokens=12,
            output_tokens=8,
        ), self.tool_calls

    def chat_json_with_lm_studio_text(self, prompt, *, system, max_tokens, temperature, context):
        self.format_prompt = prompt
        return self.payload, _Usage(input_tokens=10, output_tokens=max(1, len(json.dumps(self.payload)) // 4))

    def chat_structured_json(self, messages, *, schema_name, schema, max_tokens, temperature, context):
        prompt = messages[-1]["content"] if messages else ""
        if schema_name == "resource_query_plan":
            self.planner_prompt = prompt
            payload = self.planner_payload
        else:
            self.format_prompt = prompt
            payload = self.payload
        return payload, _Usage(input_tokens=10, output_tokens=max(1, len(json.dumps(payload)) // 4))

    def chat_json_with_lm_studio_images(self, input_items, *, system, max_tokens, temperature, context):
        self.image_input = input_items
        return self.payload, _Usage(input_tokens=12, output_tokens=max(1, len(json.dumps(self.payload)) // 4))


class FailingResourceFormatClient:
    def __init__(self):
        self.tool_prompt = None
        self.format_prompt = None

    def chat_structured_json(self, messages, *, schema_name, schema, max_tokens, temperature, context):
        prompt = messages[-1]["content"] if messages else ""
        if schema_name == "resource_query_plan":
            return {
                "queries": [{"query": "human AI collaboration resources", "intent": "Find resource candidates."}],
                "warnings": [],
            }, _Usage(input_tokens=10, output_tokens=4)
        self.format_prompt = prompt
        raise ProviderResponseError("LM Studio web resources JSON format returned no message from LM Studio.")

    def chat_text_with_lm_studio_tools(self, prompt, *, system, max_tokens, temperature, context):
        self.tool_prompt = prompt
        return (
            'brave_web_search output: [{"type":"text","text":"Title: Human-AI Collaboration: What is it and Why is it Important? | IBM\\n'
            'Description: AI can support creative generation while humans refine and evaluate its input.\\n'
            'URL: https://www.ibm.com/think/topics/human-ai-collaboration\\n\\n'
            'Title: Human-AI Co-Creativity: Exploring Synergies Across Levels of Creative Collaboration\\n'
            'Description: A research article on human and generative AI co-creative systems.\\n'
            'URL: https://arxiv.org/html/2411.12527v2"}]',
            _Usage(input_tokens=12, output_tokens=8),
            ["brave_web_search"],
        )

    def chat_json_with_lm_studio_text(self, prompt, *, system, max_tokens, temperature, context):
        self.format_prompt = prompt
        raise ProviderResponseError("LM Studio web resources JSON format returned no message from LM Studio.")


class FailingPlannerClient(FailingResourceFormatClient):
    def chat_structured_json(self, messages, *, schema_name, schema, max_tokens, temperature, context):
        if schema_name == "resource_query_plan":
            raise ProviderResponseError("planner failed")
        return super().chat_structured_json(
            messages,
            schema_name=schema_name,
            schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
            context=context,
        )


class TimeoutResourceClient:
    def chat_text_with_lm_studio_tools(self, prompt, *, system, max_tokens, temperature, context):
        raise ProviderTransientError("timed out")


class SequenceImageClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.image_inputs = []

    def chat_json_with_lm_studio_images(self, input_items, *, system, max_tokens, temperature, context):
        self.image_inputs.append(input_items)
        return self.payloads.pop(0), _Usage(input_tokens=12, output_tokens=8)


def _request(*, lecture_dir=None, transcript_text="hello local models"):
    return AnalyzeLectureRequest(
        lecture_id="lecture",
        transcript_text=transcript_text,
        segments=[{"id": 1, "text": "hello local models"}],
        slides=[{"id": 1, "relative_path": "slide.png", "linked_segment_ids": [1]}],
        duration_minutes=2.0,
        lecture_dir=lecture_dir,
    )


if __name__ == "__main__":
    unittest.main()
