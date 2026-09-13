import unittest

from scvrs.data.query_conditions import build_condition_record
from scvrs.data.query_parser import parse_query_v3


class QueryConditionRecordTest(unittest.TestCase):
    def test_subject_and_reference_fields(self):
        parsed = parse_query_v3(
            "A vehicle is a little smaller than the vehicle on the top"
        )
        record = build_condition_record(parsed)
        text = record["text_inputs"]
        self.assertEqual(text["object_text"], "vehicle")
        self.assertEqual(text["comparison_text"], "smaller_than vehicle")
        self.assertEqual(text["reference_object_text"], "vehicle")
        self.assertEqual(text["reference_position_text"], "on the top")
        self.assertIn("[COMPARISON] smaller_than vehicle", text["condition_prompt"])

    def test_simple_object_position(self):
        parsed = parse_query_v3("The baseball field on the top")
        text = build_condition_record(parsed)["text_inputs"]
        self.assertEqual(text["object_text"], "baseball field")
        self.assertEqual(text["subject_position_text"], "on the top")
        self.assertEqual(text["reference_object_text"], "")


if __name__ == "__main__":
    unittest.main()
