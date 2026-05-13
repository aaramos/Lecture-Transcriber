import unittest

from lecture_processor.slide_classifier import parse_classifier_response


class SlideClassifierTests(unittest.TestCase):
    def test_parse_classifier_response_captures_all_slide_fields(self):
        parsed = parse_classifier_response(
            "DESCRIPTION: A slide fills the frame.\n"
            "VERDICT: SLIDE\n"
            "TITLE: Questions for Business Leaders\n"
            "BUILD_STAGE: full\n"
            "LAYOUT: full-screen"
        )

        self.assertEqual("A slide fills the frame.", parsed["description"])
        self.assertEqual("SLIDE", parsed["verdict"])
        self.assertEqual("Questions for Business Leaders", parsed["title"])
        self.assertEqual("full", parsed["build_stage"])
        self.assertEqual("full-screen", parsed["layout"])

    def test_parse_classifier_response_clears_slide_fields_for_not_slide(self):
        parsed = parse_classifier_response(
            "DESCRIPTION: A person fills the frame.\n"
            "VERDICT: NOT SLIDE"
        )

        self.assertEqual("NOT SLIDE", parsed["verdict"])
        self.assertIsNone(parsed["title"])
        self.assertIsNone(parsed["build_stage"])
        self.assertIsNone(parsed["layout"])


if __name__ == "__main__":
    unittest.main()
