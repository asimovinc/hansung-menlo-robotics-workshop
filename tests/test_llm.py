import unittest

from menlo_runner.llm import build_system_prompt, get_llm_model, get_vlm_model, parse_tool_call


class ToolCallParserTest(unittest.TestCase):
    def tearDown(self):
        import os

        os.environ.pop("MENLO_LLM_MODEL", None)
        os.environ.pop("MENLO_VLM_MODEL", None)

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

    def test_get_llm_model_uses_environment_override(self):
        import os

        os.environ["MENLO_LLM_MODEL"] = "qwen/qwen3.6-35b-a3b"
        self.assertEqual(get_llm_model(), "qwen/qwen3.6-35b-a3b")

    def test_get_llm_model_rejects_unknown_model(self):
        import os

        os.environ["MENLO_LLM_MODEL"] = "not-a-real-model"
        with self.assertRaises(ValueError):
            get_llm_model()

    def test_get_vlm_model_uses_environment_override(self):
        import os

        os.environ["MENLO_VLM_MODEL"] = "minimaxai/minimax-m3"
        self.assertEqual(get_vlm_model(), "minimaxai/minimax-m3")

    def test_get_vlm_model_rejects_unknown_model(self):
        import os

        os.environ["MENLO_VLM_MODEL"] = "not-a-real-model"
        with self.assertRaises(ValueError):
            get_vlm_model()


if __name__ == "__main__":
    unittest.main()


