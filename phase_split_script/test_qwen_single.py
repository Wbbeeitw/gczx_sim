from PIL import Image

from qwen_client import call_qwen_vl

# 创建一张空白测试图
img = Image.new("RGB", (256, 256), color=(128, 128, 128))

models = [
    "qwen3-vl-flash",
    "qwen-vl-plus",
    "qwen-vl-max",
    "qwen2.5-vl-72b-instruct",
    "qwen2.5-vl-7b-instruct",
]

for model in models:
    print(f"\n=== Testing model: {model} ===")
    try:
        result = call_qwen_vl(
            images=[img],
            prompt='What is in this image? Reply with a JSON object: {"description": "brief text"}',
            model=model,
            max_retries=1,
        )
        print("OK:", result)
    except Exception as e:
        print("FAILED:", type(e).__name__, e)
