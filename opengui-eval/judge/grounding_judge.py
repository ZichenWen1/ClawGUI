"""
Grounding judge for ScreenSpot-style benchmarks.
Evaluates whether a predicted click point falls inside the ground-truth bounding box.
"""
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from base_judge import BaseJudge


def qwen3vl_parse(infer_str: str, image_size: List[int]) -> Optional[Tuple[float, float]]:
    """
    Parse Qwen3-VL output into absolute pixel coordinates.

    Output formats:
      1. Tool-call JSON: <tool_call>{"name": "computer_use", "arguments": {"coordinate": [x, y]}}</tool_call>
      2. Bracket list:   "[x, y]" or "[x1, y1, x2, y2]"

    Coordinates are [0, 1000]-normalized and converted to absolute pixels.

    Args:
        infer_str: Raw model output string.
        image_size: [width, height] of the original image.

    Returns:
        (x, y) absolute pixel coordinates, or None if parsing fails.
    """
    if not infer_str:
        return None

    coords = None

    # Method 1: parse <tool_call> JSON format
    if '<tool_call>' in infer_str and '</tool_call>' in infer_str:
        try:
            start_idx = infer_str.find('<tool_call>') + len('<tool_call>')
            end_idx = infer_str.find('</tool_call>')
            json_str = infer_str[start_idx:end_idx].strip()
            tool_call_json = json.loads(json_str)
            if 'arguments' in tool_call_json and 'coordinate' in tool_call_json['arguments']:
                coordinate = tool_call_json['arguments']['coordinate']
                if isinstance(coordinate, list) and len(coordinate) >= 2:
                    coords = [float(coordinate[0]), float(coordinate[1])]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError):
            pass

    # Method 2: fallback — extract numbers with regex
    if coords is None:
        numbers = re.findall(r'-?\d+\.?\d*', infer_str)
        if numbers:
            coords = [float(n) for n in numbers]

    if not coords:
        return None

    width, height = image_size

    if len(coords) == 2:
        # Point: [0, 1000]-normalized → absolute
        return (int(coords[0] / 1000 * width), int(coords[1] / 1000 * height))
    elif len(coords) >= 4:
        # BBox: use center point, [0, 1000]-normalized → absolute
        x_center = (coords[0] + coords[2]) / 2
        y_center = (coords[1] + coords[3]) / 2
        return (int(x_center / 1000 * width), int(y_center / 1000 * height))

    return None


def qwen25vl_parse(infer_str: str, image_size: List[int]) -> Optional[Tuple[float, float]]:
    """
    Parse Qwen2.5-VL output into absolute pixel coordinates.

    Qwen2.5-VL outputs absolute pixel coordinates directly (no normalization needed).

    Output formats: same as qwen3vl_parse.
    """
    if not infer_str:
        return None

    coords = None

    if '<tool_call>' in infer_str and '</tool_call>' in infer_str:
        try:
            start_idx = infer_str.find('<tool_call>') + len('<tool_call>')
            end_idx = infer_str.find('</tool_call>')
            json_str = infer_str[start_idx:end_idx].strip()
            tool_call_json = json.loads(json_str)
            if 'arguments' in tool_call_json and 'coordinate' in tool_call_json['arguments']:
                coordinate = tool_call_json['arguments']['coordinate']
                if isinstance(coordinate, list) and len(coordinate) >= 2:
                    coords = [float(coordinate[0]), float(coordinate[1])]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError):
            pass

    if coords is None:
        numbers = re.findall(r'-?\d+\.?\d*', infer_str)
        if numbers:
            coords = [float(n) for n in numbers]

    if not coords:
        return None

    if len(coords) == 2:
        # Absolute coordinates
        return (int(coords[0]), int(coords[1]))
    elif len(coords) >= 4:
        # BBox: use center point (absolute coordinates)
        x_center = (coords[0] + coords[2]) / 2
        y_center = (coords[1] + coords[3]) / 2
        return (x_center, y_center)

    return None


def stepgui_parse(infer_str: str, image_size: List[int]) -> Optional[Tuple[float, float]]:
    """
    Parse StepGUI output: "point:107,611"
    Coordinates are [0, 999]-normalized.
    """
    if not infer_str:
        return None

    numbers = re.findall(r'-?\d+\.?\d*', infer_str)
    if len(numbers) < 2:
        return None

    coords = [float(numbers[0]), float(numbers[1])]
    width, height = image_size
    return (int(coords[0] / 999 * width), int(coords[1] / 999 * height))


def uitars_parse(infer_str: str, image_size: List[int]) -> Optional[Tuple[float, float]]:
    """
    Parse UI-TARS output and map back to original image coordinates.

    UI-TARS outputs absolute pixel coordinates in the smart_resize'd image space.
    This function reverses the smart_resize transform to get original-image coordinates.

    Output format: "Action: click(point='<point>197 525</point>')"

    Args:
        infer_str: Raw model output string.
        image_size: [width, height] of the original image.

    Returns:
        (x, y) absolute pixel coordinates in the original image, or None.
    """
    import math

    if not infer_str:
        return None

    # Extract coordinates from model output
    point_match = re.search(r"<point>(\d+)\s+(\d+)</point>", infer_str)
    if point_match:
        model_x = int(point_match.group(1))
        model_y = int(point_match.group(2))
    else:
        coord_match = re.search(r"\((\d+),\s*(\d+)\)", infer_str)
        if coord_match:
            model_x = int(coord_match.group(1))
            model_y = int(coord_match.group(2))
        else:
            return None

    # Reproduce the smart_resize transform used by Qwen2.5-VL / UI-TARS
    orig_w, orig_h = image_size
    IMAGE_FACTOR = 28
    MIN_PIXELS = 100 * 28 * 28    # 78400
    MAX_PIXELS = 16384 * 28 * 28  # 12845056

    def round_by_factor(n, f): return round(n / f) * f
    def floor_by_factor(n, f): return math.floor(n / f) * f
    def ceil_by_factor(n, f):  return math.ceil(n / f) * f

    h_bar = max(IMAGE_FACTOR, round_by_factor(orig_h, IMAGE_FACTOR))
    w_bar = max(IMAGE_FACTOR, round_by_factor(orig_w, IMAGE_FACTOR))
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((orig_h * orig_w) / MAX_PIXELS)
        h_bar = floor_by_factor(int(orig_h / beta), IMAGE_FACTOR)
        w_bar = floor_by_factor(int(orig_w / beta), IMAGE_FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (orig_h * orig_w))
        h_bar = ceil_by_factor(int(orig_h * beta), IMAGE_FACTOR)
        w_bar = ceil_by_factor(int(orig_w * beta), IMAGE_FACTOR)

    smart_h, smart_w = h_bar, w_bar

    # Map smart_resize coordinates back to original image coordinates
    abs_x = int(model_x / smart_w * orig_w)
    abs_y = int(model_y / smart_h * orig_h)
    return (abs_x, abs_y)


def uivenus15_parse(infer_str: str, image_size: List[int]) -> Optional[Tuple[float, float]]:
    """
    Parse UI-Venus 1.5 output into absolute pixel coordinates.

    Output format: "[x, y]" or "[x1, y1, x2, y2]"
    Coordinates are [0, 1000]-normalized. Special value [-1, -1] signals infeasibility.
    """
    if not infer_str:
        return None

    infer_str = infer_str.strip()

    # Infeasibility signal
    if '[-1' in infer_str and '-1]' in infer_str:
        numbers = re.findall(r'-?\d+\.?\d*', infer_str)
        if numbers and all(float(n) == -1 for n in numbers[:2]):
            return None

    numbers = re.findall(r'-?\d+\.?\d*', infer_str)
    if not numbers:
        return None

    coords = [float(n) for n in numbers]
    width, height = image_size

    if len(coords) == 2:
        return (coords[0] / 1000 * width, coords[1] / 1000 * height)
    elif len(coords) >= 4:
        x_center = (coords[0] + coords[2]) / 2
        y_center = (coords[1] + coords[3]) / 2
        return (x_center / 1000 * width, y_center / 1000 * height)

    return None


def maiui_parse(infer_str: str, image_size: List[int]) -> Optional[Tuple[float, float]]:
    """
    Parse MAI-UI output into absolute pixel coordinates.

    Output format: <answer>{"coordinate": [73, 19]}</answer>
    Coordinates are [0, 1000]-normalized (divide by 1000 to map to image size).
    """
    if not infer_str:
        return None

    coords = None

    if '<answer>' in infer_str and '</answer>' in infer_str:
        try:
            start_idx = infer_str.find('<answer>') + len('<answer>')
            end_idx = infer_str.find('</answer>')
            json_str = infer_str[start_idx:end_idx].strip()
            answer_json = json.loads(json_str)
            if 'coordinate' in answer_json:
                coordinate = answer_json['coordinate']
                if isinstance(coordinate, list) and len(coordinate) >= 2:
                    coords = [float(coordinate[0]), float(coordinate[1])]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError):
            pass

    if coords is None:
        numbers = re.findall(r'-?\d+\.?\d*', infer_str)
        if numbers:
            coords = [float(n) for n in numbers]

    if not coords:
        return None

    width, height = image_size

    if len(coords) == 2:
        return (int(coords[0] / 1000 * width), int(coords[1] / 1000 * height))
    elif len(coords) >= 4:
        x_center = (coords[0] + coords[2]) / 2
        y_center = (coords[1] + coords[3]) / 2
        return (int(x_center / 1000 * width), int(y_center / 1000 * height))

    return None


class ScreenSpotJudge(BaseJudge):
    """Judge for ScreenSpot-style grounding benchmarks (point-in-box evaluation)."""

    def __init__(self, benchmark_name: str = "screenspot-pro"):
        super().__init__(benchmark_name)

    def parse_prediction(self, item: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        Dispatch to the appropriate parse function based on model_type detected
        from the {model_type}_infer key in the item.

        Returns:
            (x, y) absolute pixel coordinates, or None.
        """
        image_size = item.get('image_size')
        if not image_size:
            return None

        for key, value in item.items():
            if key.endswith('_infer'):
                model_type = key.replace('_infer', '')
                if model_type in ('qwen3vl', 'guiowl15'):
                    return qwen3vl_parse(value, image_size)
                elif model_type in ('qwen25vl', 'guig2'):
                    return qwen25vl_parse(value, image_size)
                elif model_type == 'uitars':
                    return uitars_parse(value, image_size)
                elif model_type == 'stepgui':
                    return stepgui_parse(value, image_size)
                elif model_type == 'uivenus15':
                    return uivenus15_parse(value, image_size)
                elif model_type == 'maiui':
                    return maiui_parse(value, image_size)
                else:
                    supported = ['qwen3vl', 'guiowl15', 'qwen25vl', 'guig2', 'uitars', 'stepgui', 'uivenus15', 'maiui']
                    raise ValueError(f"Unsupported model_type: '{model_type}'. Supported: {supported}")
        return None

    def evaluate_single(self, pred: Optional[Tuple[float, float]], gt: List[float]) -> bool:
        """
        Check whether the predicted point falls inside the ground-truth bounding box.

        Args:
            pred: (x, y) predicted click point in absolute pixels.
            gt: [x1, y1, x2, y2] ground-truth bounding box in absolute pixels.

        Returns:
            True if pred is inside gt.
        """
        if pred is None:
            return False
        x, y = pred
        x1, y1, x2, y2 = gt
        return x1 <= x <= x2 and y1 <= y <= y2


def main():
    parser = argparse.ArgumentParser(description='Grounding judge for ScreenSpot-style benchmarks')

    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to predictions file (JSON/JSONL)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to write judged output')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--benchmark', type=str, default='screenspot-pro',
                        help='Benchmark name')
    parser.add_argument('--model_type', type=str, default=None,
                        help='Model type override (auto-detected if not provided)')

    args = parser.parse_args()

    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    judge = ScreenSpotJudge(benchmark_name=args.benchmark)
    judge.evaluate(
        input_file=args.input_file,
        output_file=args.output_file,
        exp_name=args.exp_name,
        model_type=args.model_type
    )


if __name__ == '__main__':
    main()
