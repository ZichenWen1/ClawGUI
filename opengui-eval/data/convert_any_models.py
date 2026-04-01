"""
Inject model-specific prompts into benchmark JSON files.

Each input JSON must have a "question" field per item.
For osworld-g, uivenus15 and guiowl15 automatically use refusal-aware prompts.

Usage:
    python data/convert_any_models.py --input data/osworld-g.json
    python data/convert_any_models.py --input data/osworld-g.json --models stepgui guig2
    python data/convert_any_models.py --input data/osworld-g.json data/uivision.json
"""
import json
import copy
import argparse
from pathlib import Path

# =============================================================================
# Prompt constants
# =============================================================================

# Used by qwen3vl. Resolution hardcoded to 1000x1000 (normalized coordinate space).
COMPUTER_USE_SYSTEM_PROMPT = (
    "You are a helpful assistant. The user will give you an instruction, and you MUST left click on the "
    "corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    "{\"type\": \"function\", \"function\": {\"name\": \"computer_use\", \"description\": "
    "\"Use a mouse to interact with a computer.\\n"
    "* The screen's resolution is 1000x1000.\\n"
    "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \\n"
    "* You can only use the left_click action to interact with the computer.\", "
    "\"parameters\": {\"properties\": {\"action\": {\"description\": "
    "\"The action to perform. The available actions are:\\n"
    "* `left_click`: Click the left mouse button with coordinate (x, y).\", "
    "\"enum\": [\"left_click\"], \"type\": \"string\"}, "
    "\"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
    "coordinates to move the mouse to. Required only by `action=left_click`.\", \"type\": \"array\"}, "
    "\"required\": [\"action\"], \"type\": \"object\"}}}\n"
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
    "</tool_call>"
)

# Used by qwen25vl. Resolution is filled in per-item from image_size via build_qwen25vl_system_prompt().
QWEN25VL_SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant. The user will give you an instruction, and you MUST left click on the "
    "corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    "{\"type\": \"function\", \"function\": {\"name\": \"computer_use\", \"description\": "
    "\"Use a mouse to interact with a computer.\\n"
    "* The screen's resolution is __RESOLUTION__.\\n"
    "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. "
    "Don't click boxes on their edges unless asked.\\n"
    "* you can only use the left_click and mouse_move action to interact with the computer. "
    "if you can't find the element, you should terminate the task and report the failure.\", "
    "\"parameters\": {\"properties\": {\"action\": {\"description\": "
    "\"The action to perform. The available actions are:\\n"
    "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n"
    "* `left_click`: Click the left mouse button with coordinate (x, y).\", "
    "\"type\": \"string\"}, "
    "\"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
    "coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click`.\", "
    "\"type\": \"array\"}, "
    "\"status\": {\"description\": \"The status of the task. Required only by `action=terminate`.\", "
    "\"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}, "
    "\"required\": [\"action\"], \"type\": \"object\"}}}\n"
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
    "</tool_call>\n"
)


def build_qwen25vl_system_prompt(width: int, height: int) -> str:
    return QWEN25VL_SYSTEM_PROMPT_TEMPLATE.replace("__RESOLUTION__", f"{width}x{height}")


UITARS_GROUNDING_PROMPT = (
    "You are a GUI agent. You are given a task and your action history, with screenshots. "
    "You need to perform the next action to complete the task. \n\n"
    "## Output Format\n\nAction: ...\n\n\n"
    "## Action Space\nclick(point='<point>x1 y1</point>'')\n\n"
    "## User Instruction\n{instruction}"
)

MAIUI_GROUNDING_SYSTEM_PROMPT = """You are a GUI grounding agent. 
## Task
Given a screenshot and the user's grounding instruction. Your task is to accurately locate a UI element based on the user's instructions.
First, you should carefully examine the screenshot and analyze the user's instructions,  translate the user's instruction into a effective reasoning process, and then provide the final coordinate.
## Output Format
Return a json object with a reasoning process in <grounding_think></grounding_think> tags, a [x,y] format coordinate within <answer></answer> XML tags:
<grounding_think>...</grounding_think>
<answer>
{"coordinate": [x,y]}
</answer>"""

UIVENUS15_GROUNDING_PROMPT = (
    "Output the center point of the position corresponding to the following instruction: \n"
    "{}. \n\n"
    "The output should just be the coordinates of a point, in the format [x,y]."
)

GUIOWL15_SYSTEM_PROMPT = (
    '# Tools\n\n'
    'You may call one or more functions to assist with the user query.\n\n'
    'You are provided with function signatures within <tools></tools> XML tags:\n'
    '<tools>\n'
    '{"type": "function", "function": {"name": "computer_use", '
    '"description": "Use a mouse to interact with a computer.\n'
    "* The screen's resolution is 1000x1000.\n"
    '* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. '
    "Don't click boxes on their edges unless asked.\n"
    "* don't use any other computer use tool like type, key, scroll, left_click_drag and so on.\n"
    '* you can only use the left_click and mouse_move action to interact with the computer. '
    'if you can\'t find the element, you should terminate the task and report the failure.", '
    '"parameters": {"properties": {"action": {"description": '
    '"The action to perform. The available actions are:\n'
    '* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n'
    '* `left_click`: Click the left mouse button with coordinate (x, y) pixel coordinate on the screen.", '
    '"enum": ["mouse_move", "left_click"], "type": "string"}, '
    '"coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) '
    'coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click`.", '
    '"type": "array"}}, "required": ["action"], "type": "object"}}}\n'
    '</tools>\n\n'
    'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
    '<tool_call>\n'
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    '</tool_call>\n'
)

STEPGUI_SUFFIX = "\noutput its center point coordinates that fit 0-1000 range like point:x_center,y_center"

GUIG2_GROUNDING_PROMPT = "Outline the position corresponding to the instruction: {instruction}. The output should be only [x1,y1,x2,y2]."

# Appended to uivenus15 question for osworld-g (infeasible task signal).
UIVENUS15_REFUSAL_SUFFIX = (
    " Additionally, if the task is infeasible "
    "(e.g., the task is not related to the image), the output should be [-1,-1]."
)

# Appended to guiowl15 system prompt for osworld-g (infeasible task signal).
GUIOWL15_INFEASIBLE_PREFIX = (
    "\nAdditionally, if you think the task is infeasible "
    "(e.g., the task is not related to the image), return "
    "<tool_call>\n"
    '{\"name\": \"computer_use\", \"arguments\": {\"action\": \"terminate\", \"status\": \"failure\"}}\n'
    "</tool_call>"
)


# =============================================================================
# Conversion functions (all return a deep copy; input data is never modified)
# =============================================================================

def convert_qwen3vl(data):
    out = copy.deepcopy(data)
    for item in out:
        item['system_prompt'] = [COMPUTER_USE_SYSTEM_PROMPT]
    return out


def convert_qwen25vl(data):
    out = copy.deepcopy(data)
    for item in out:
        image_size = item.get('image_size', [1000, 1000])
        width, height = int(image_size[0]), int(image_size[1])
        item['system_prompt'] = [build_qwen25vl_system_prompt(width, height)]
    return out


def convert_uitars(data):
    out = copy.deepcopy(data)
    for item in out:
        item['question'] = UITARS_GROUNDING_PROMPT.format(instruction=item['question'])
    return out


def convert_maiui(data):
    out = copy.deepcopy(data)
    for item in out:
        item['system_prompt'] = [MAIUI_GROUNDING_SYSTEM_PROMPT]
    return out


def convert_uivenus15(data):
    out = copy.deepcopy(data)
    for item in out:
        q = item['question']
        if q.endswith('.'):
            q = q[:-1]
        item['question'] = UIVENUS15_GROUNDING_PROMPT.format(q)
    return out


def convert_uivenus15_refusal(data):
    out = copy.deepcopy(data)
    for item in out:
        q = item['question']
        if q.endswith('.'):
            q = q[:-1]
        item['question'] = UIVENUS15_GROUNDING_PROMPT.format(q) + UIVENUS15_REFUSAL_SUFFIX
    return out


def convert_guiowl15(data):
    out = copy.deepcopy(data)
    for item in out:
        item['system_prompt'] = [GUIOWL15_SYSTEM_PROMPT]
    return out


def convert_guiowl15_refusal(data):
    out = copy.deepcopy(data)
    for item in out:
        item['system_prompt'] = [GUIOWL15_SYSTEM_PROMPT + GUIOWL15_INFEASIBLE_PREFIX]
    return out


def convert_stepgui(data):
    out = copy.deepcopy(data)
    for item in out:
        item['question'] = item['question'] + STEPGUI_SUFFIX
    return out


def convert_guig2(data):
    out = copy.deepcopy(data)
    for item in out:
        q = item['question']
        if q.endswith('.'):
            q = q[:-1]
        item['question'] = GUIG2_GROUNDING_PROMPT.format(instruction=q)
    return out


REGISTRY = {
    "qwen3vl":   convert_qwen3vl,
    "qwen25vl":  convert_qwen25vl,
    "uitars":    convert_uitars,
    "maiui":     convert_maiui,
    "uivenus15": convert_uivenus15,
    "guiowl15":  convert_guiowl15,
    "stepgui":   convert_stepgui,
    "guig2":     convert_guig2,
}

ALL_MODELS = list(REGISTRY.keys())

# For osworld-g, override uivenus15/guiowl15 with refusal-aware variants.
# Output filenames remain unchanged (no -refusal suffix).
_REFUSAL_OVERRIDES: dict = {
    "osworld-g": {
        "uivenus15": convert_uivenus15_refusal,
        "guiowl15":  convert_guiowl15_refusal,
    },
}


# =============================================================================
# Auto-registration into main.py
# =============================================================================

MAIN_PY = Path(__file__).parent.parent / 'main.py'
GUI_GROUNDING_ROOT = Path(__file__).parent.parent  # opengui-eval/


def register_in_main_py(new_entries: dict):
    """Append new {key: path} entries to BENCHMARK_DATA_MAP in main.py; skip existing keys."""
    if not MAIN_PY.exists():
        print(f"  [skip registration] main.py not found: {MAIN_PY}")
        return

    text = MAIN_PY.read_text(encoding='utf-8')

    import re
    map_start = text.find('BENCHMARK_DATA_MAP = {')
    if map_start == -1:
        print("  [skip registration] BENCHMARK_DATA_MAP not found")
        return

    existing_keys = set(re.findall(r'"([^"]+)"\s*:', text[map_start:]))
    to_add = {k: v for k, v in new_entries.items() if k not in existing_keys}
    if not to_add:
        return

    map_end = text.find('\n}', map_start)
    if map_end == -1:
        print("  [skip registration] BENCHMARK_DATA_MAP closing brace not found")
        return

    pad = 44
    lines = []
    for key, rel_path in sorted(to_add.items()):
        quoted_key = f'"{key}"'
        lines.append(f'    {quoted_key:<{pad}}: "{rel_path}",')
    insert_str = '\n' + '\n'.join(lines)

    text = text[:map_end] + insert_str + text[map_end:]
    MAIN_PY.write_text(text, encoding='utf-8')

    for key, rel_path in sorted(to_add.items()):
        print(f"  [registered] {key:<44} -> {rel_path}")


# =============================================================================
# Core processing
# =============================================================================

def process_file(input_path: Path, output_dir: Path, models: list):
    print(f"\n{'='*60}")
    print(f"input: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    print(f"loaded {len(base_data)} items")

    missing = [i for i, d in enumerate(base_data) if 'question' not in d]
    if missing:
        print(f"  [WARNING] {len(missing)} items missing 'question' field, skipping file")
        return

    stem = input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    refusal_map: dict = {}
    for prefix, overrides in _REFUSAL_OVERRIDES.items():
        if stem == prefix or stem.startswith(prefix + "-"):
            refusal_map = overrides
            break

    new_entries = {}
    for model in models:
        fn = refusal_map.get(model) or REGISTRY[model]
        converted = fn(base_data)
        out_path = output_dir / f"{stem}-{model}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        suffix_note = " [refusal]" if model in refusal_map else ""
        print(f"  [{model:<10}] -> {out_path}{suffix_note}")

        try:
            rel_path = out_path.resolve().relative_to(GUI_GROUNDING_ROOT.resolve())
            new_entries[f"{stem}-{model}"] = str(rel_path)
        except ValueError:
            new_entries[f"{stem}-{model}"] = str(out_path.resolve())

    register_in_main_py(new_entries)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Inject model-specific prompts into benchmark JSON files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', nargs='+', required=True, help='Input JSON file(s)')
    parser.add_argument(
        '--models', nargs='+', default=ALL_MODELS,
        choices=ALL_MODELS,
        help=f'Models to inject (default: all). Options: {ALL_MODELS}'
    )
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: same as input)')

    args = parser.parse_args()
    print(f"models: {args.models}")

    for input_str in args.input:
        input_path = Path(input_str)
        if not input_path.exists():
            print(f"[ERROR] file not found: {input_path}")
            continue
        out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
        process_file(input_path, out_dir, args.models)

    print(f"\n{'='*60}")
    print("done.")


if __name__ == '__main__':
    main()
