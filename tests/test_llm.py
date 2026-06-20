import unittest

from menlo_runner.llm import build_system_prompt, parse_tool_call


class ToolCallParserTest(unittest.TestCase):
    def test_parse_fenced_json(self):
        text = '```json\n{"tool": "set_velocity", "args": {"vx": 0.8}}\n```'
        self.assertEqual(
            parse_tool_call(text),
            {"tool": "set_velocity", "args": {"vx": 0.8}},
        )

    def test_parse_unfenced_json(self):
        text = '{"tool": "done", "args": {"summary": "finished"}}'
        self.assertEqual(
            parse_tool_call(text),
            {"tool": "done", "args": {"summary": "finished"}},
        )

    def test_missing_args_defaults_to_empty_dict(self):
        self.assertEqual(parse_tool_call('{"tool": "look"}'), {"tool": "look", "args": {}})

    def test_invalid_reply_returns_none(self):
        self.assertIsNone(parse_tool_call("no tool call here"))

    def test_system_prompt_lists_tools(self):
        prompt = build_system_prompt({"look": {"description": "Take a picture."}})
        self.assertIn("look", prompt)
        self.assertIn("Take a picture.", prompt)


if __name__ == "__main__":
    unittest.main()


