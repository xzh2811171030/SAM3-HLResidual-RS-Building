import cv2
from PIL import Image
from pathlib import Path
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DEVICE = "cuda"
checkpoint = Path("./weights/sam3.pt")  # 你的实际路径
img_path = "./data/raw/whu_mix/train/image/xxx.png"  # 替换为真实存在的图片  # 替换为真实存在的图片

model = build_sam3_image_model(str(checkpoint))
model.to(DEVICE).eval()
processor = Sam3Processor(model, confidence_threshold=0.2)

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil = Image.fromarray(img_rgb)

state = processor.set_image(pil)
state = processor.set_text_prompt(state, prompt="building")
masks, scores, logits = processor.predict_masks(state)   # 关键调用

print("masks type:", type(masks))
if masks is not None:
    print("masks length:", len(masks))
    if len(masks) > 0:
        print("mask shape:", masks[0].shape)
else:
    print("No masks generated")