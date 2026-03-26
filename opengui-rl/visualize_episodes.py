"""
可视化 episode 中的 click/swipe 动作到对应的截图上。
- click: 在图片上画红点
- swipe: 在图片上画红色箭头（起点到终点）
输出保存到每个 episode 文件夹下的 marker_images/ 目录。
"""

import json
import os
import re
import glob
from PIL import Image, ImageDraw

EPISODE_ROOT = "/home/tangfei/online_rl/verl-agent/episode"
MARKER_RADIUS = 12  # 红点半径
ARROW_WIDTH = 4     # 箭头线宽


def parse_action(model_response: str):
    """从 model_response 中解析出 action 和相关参数。"""
    # 提取 <tool_call> 标签内的 JSON
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', model_response, re.DOTALL)
    if not match:
        return None
    try:
        call = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    args = call.get("arguments", {})
    return args


def draw_click(draw: ImageDraw.ImageDraw, x: int, y: int):
    """在 (x, y) 处画一个红色实心圆点。"""
    r = MARKER_RADIUS
    draw.ellipse([x - r, y - r, x + r, y + r], fill="red", outline="red")


def draw_swipe(draw: ImageDraw.ImageDraw, start: list, end: list):
    """画一条从 start 到 end 的红色箭头线。"""
    x0, y0 = start
    x1, y1 = end
    draw.line([(x0, y0), (x1, y1)], fill="red", width=ARROW_WIDTH)
    # 在终点画一个小圆点表示方向
    r = MARKER_RADIUS // 2
    draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill="red")
    # 在起点画一个绿色小圆点区分起止
    draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill="green")


def process_episode(episode_json_path: str):
    """处理单个 episode.json，生成标注图片。"""
    episode_dir = os.path.dirname(episode_json_path)
    marker_dir = os.path.join(episode_dir, "marker_images")
    os.makedirs(marker_dir, exist_ok=True)

    with open(episode_json_path, "r") as f:
        data = json.load(f)

    task_name = data.get("meta_data", {}).get("task_name", "unknown")
    episode_id = data.get("meta_data", {}).get("episode_id", "unknown")
    print(f"Processing: {task_name} / {episode_id}")

    for step_data in data.get("episode", []):
        step = step_data["step"]
        image_path = step_data["image_path"]
        model_response = step_data.get("model_response", "")

        if not os.path.exists(image_path):
            print(f"  [WARN] Image not found: {image_path}")
            continue

        args = parse_action(model_response)
        if args is None:
            continue

        action = args.get("action", "")
        img = Image.open(image_path).copy()
        draw = ImageDraw.Draw(img)
        marked = False

        if action == "click" and "coordinate" in args:
            coord = args["coordinate"]
            draw_click(draw, coord[0], coord[1])
            marked = True

        elif action == "swipe" and "startCoordinate" in args and "endCoordinate" in args:
            draw_swipe(draw, args["startCoordinate"], args["endCoordinate"])
            marked = True

        elif action == "swipe" and "coordinate" in args and "endCoordinate" in args:
            draw_swipe(draw, args["coordinate"], args["endCoordinate"])
            marked = True

        if marked:
            out_path = os.path.join(marker_dir, f"step_{step:04d}.png")
            img.save(out_path)
            print(f"  Saved: step_{step:04d}.png  ({action})")
        else:
            # 非 click/swipe 动作，也拷贝原图方便查看完整序列
            out_path = os.path.join(marker_dir, f"step_{step:04d}.png")
            img.save(out_path)
            print(f"  Saved: step_{step:04d}.png  (no marker, action={action})")


def main():
    # 遍历所有任务目录下的 episode.json
    pattern = os.path.join(EPISODE_ROOT, "*", "*", "episode.json")
    episode_files = sorted(glob.glob(pattern))

    if not episode_files:
        print(f"No episode.json found under {EPISODE_ROOT}")
        return

    print(f"Found {len(episode_files)} episode(s)\n")
    for ep_file in episode_files:
        process_episode(ep_file)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
