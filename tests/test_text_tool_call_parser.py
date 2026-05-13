"""Tests for `_parse_text_tool_calls` — text-channel tool-call recovery.

Different model families emit tool calls in different formats when oMLX
doesn't promote them to structured `tool_calls`. The parser must accept:

  Qwen/Hermes:    <tool_call>{"name":...,"arguments":...}</tool_call>
  Qwen2.5-Coder:  bare JSON {"name":...,"arguments":...}
  Llama-3.x:      {"type":"function","name":...,"parameters":...}
                  (or bare {"name":...,"parameters":...})
"""

from llamabench.agents.loop import _parse_text_tool_calls

KNOWN = {"list_directory", "read_file"}
KNOWN_DOTTED = {
    "triangle_properties.get",
    "spotify.play",
    "math_toolkit.sum_of_multiples",
    "math_toolkit.product_of_primes",
}


def test_qwen_tagged_format():
    text = (
        'Sure, I will list it.\n'
        '<tool_call>{"name": "list_directory", "arguments": {"path": "/tmp"}}</tool_call>'
    )
    calls = _parse_text_tool_calls(text, KNOWN)
    assert len(calls) == 1
    assert calls[0].name == "list_directory"
    assert calls[0].arguments == {"path": "/tmp"}


def test_qwen_coder_bare_json():
    text = '{"name": "list_directory", "arguments": {"path": "/var/folders/x"}}'
    calls = _parse_text_tool_calls(text, KNOWN)
    assert len(calls) == 1
    assert calls[0].name == "list_directory"
    assert calls[0].arguments == {"path": "/var/folders/x"}


def test_llama_with_type_field():
    text = '{"type": "function", "name": "list_directory", "parameters": {"path": "/tmp"}}'
    calls = _parse_text_tool_calls(text, KNOWN)
    assert len(calls) == 1
    assert calls[0].name == "list_directory"
    assert calls[0].arguments == {"path": "/tmp"}


def test_llama_bare_parameters():
    text = '{"name": "read_file", "parameters": {"path": "/tmp/a.txt"}}'
    calls = _parse_text_tool_calls(text, KNOWN)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "/tmp/a.txt"}


def test_unknown_tool_name_rejected():
    text = '{"name": "rm_rf", "arguments": {"path": "/"}}'
    calls = _parse_text_tool_calls(text, KNOWN)
    assert calls == []


def test_no_tool_call_in_text():
    text = "I cannot fulfill that request — I do not have filesystem access."
    calls = _parse_text_tool_calls(text, KNOWN)
    assert calls == []


def test_returns_all_emitted_calls():
    # Parallel BFCL categories require every emitted call. Agent loops
    # dispatch multiple calls in order, so accumulating is correct for both.
    text = (
        '{"name": "list_directory", "arguments": {"path": "/a"}}\n'
        '{"name": "read_file", "arguments": {"path": "/a/x"}}'
    )
    calls = _parse_text_tool_calls(text, KNOWN)
    assert len(calls) == 2
    assert calls[0].arguments == {"path": "/a"}
    assert calls[1].name == "read_file"
    assert calls[1].arguments == {"path": "/a/x"}


def test_thinking_preamble_then_call():
    # Qwen3 reasoning models emit <think>...</think> first.
    text = (
        "<think>The user wants the listing of /tmp. I should call list_directory.</think>\n"
        "<tool_call>{\"name\": \"list_directory\", \"arguments\": {\"path\": \"/tmp\"}}</tool_call>"
    )
    calls = _parse_text_tool_calls(text, KNOWN)
    assert len(calls) == 1
    assert calls[0].name == "list_directory"


def test_arguments_as_string_alias():
    # Some models double-encode arguments as a JSON string.
    text = '<tool_call>{"name": "list_directory", "arguments": "{\\"path\\": \\"/tmp\\"}"}</tool_call>'
    calls = _parse_text_tool_calls(text, KNOWN)
    assert len(calls) == 1
    assert calls[0].arguments == {"path": "/tmp"}


def test_llama_array_of_strings():
    # Observed on Llama-3.3-70B-Instruct-3bit's 0d adversarial probe:
    # the model wraps each tool call as a JSON-encoded string and emits
    # them inside a JSON array.
    text = (
        '["{\\"type\\": \\"function\\", \\"name\\": \\"read_file\\", '
        '\\"parameters\\": {\\"path\\": \\"/tmp/b.txt\\"}}", '
        '"{\\"type\\": \\"function\\", \\"name\\": \\"read_file\\", '
        '\\"parameters\\": {\\"path\\": \\"/tmp/z.txt\\"}}"]'
    )
    calls = _parse_text_tool_calls(text, KNOWN)
    assert len(calls) == 2
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "/tmp/b.txt"}
    assert calls[1].arguments == {"path": "/tmp/z.txt"}


def test_llama_array_of_dicts():
    # Same outer form but inner elements already parsed as dicts (some
    # samplers emit this shape directly).
    text = (
        '[{"type": "function", "name": "list_directory", "parameters": {"path": "/a"}}]'
    )
    calls = _parse_text_tool_calls(text, KNOWN)
    assert len(calls) == 1
    assert calls[0].name == "list_directory"
    assert calls[0].arguments == {"path": "/a"}


def test_array_with_unknown_then_known():
    # Skip an unknown name and pick the first known one.
    text = (
        '["{\\"name\\": \\"rm_rf\\", \\"parameters\\": {\\"path\\": \\"/\\"}}", '
        '"{\\"name\\": \\"list_directory\\", \\"parameters\\": {\\"path\\": \\"/tmp\\"}}"]'
    )
    calls = _parse_text_tool_calls(text, KNOWN)
    assert len(calls) == 1
    assert calls[0].name == "list_directory"


def test_dotted_function_name():
    # BFCL "multiple"/"parallel" use namespaced names like
    # "triangle_properties.get". Old `\\w+` regex rejected the dot.
    text = '{"name": "triangle_properties.get", "arguments": {"side1": 5, "side2": 4, "side3": 3}}'
    calls = _parse_text_tool_calls(text, KNOWN_DOTTED)
    assert len(calls) == 1
    assert calls[0].name == "triangle_properties.get"
    assert calls[0].arguments == {"side1": 5, "side2": 4, "side3": 3}


def test_parallel_calls_with_prose_preamble():
    # BFCL "parallel" emission shape (Qwen2.5-Coder): prose explanation
    # followed by N separate JSON blobs, all of which must be returned.
    text = (
        "To achieve this, I will call the `spotify.play` function twice.\n\n"
        "First, for Taylor Swift:\n"
        '{"name": "spotify.play", "arguments": {"artist": "Taylor Swift", "duration": 20}}\n\n'
        "Then, for Maroon 5:\n"
        '{"name": "spotify.play", "arguments": {"artist": "Maroon 5", "duration": 15}}'
    )
    calls = _parse_text_tool_calls(text, KNOWN_DOTTED)
    assert len(calls) == 2
    assert calls[0].arguments == {"artist": "Taylor Swift", "duration": 20}
    assert calls[1].arguments == {"artist": "Maroon 5", "duration": 15}


def test_tools_wrapper_with_multiple_calls():
    # BFCL "parallel_multiple" emission shape (Qwen2.5-Coder): multiple
    # bare-JSON objects wrapped in a single <tools>...</tools> block.
    text = (
        "<tools>\n"
        '{"name": "math_toolkit.sum_of_multiples", "arguments": {"lower_limit": 1, "upper_limit": 1000, "multiples": [3, 5]}}\n'
        '{"name": "math_toolkit.product_of_primes", "arguments": {"count": 5}}\n'
        "</tools>"
    )
    calls = _parse_text_tool_calls(text, KNOWN_DOTTED)
    assert len(calls) == 2
    assert calls[0].name == "math_toolkit.sum_of_multiples"
    assert calls[1].name == "math_toolkit.product_of_primes"
