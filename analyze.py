import os
import io
import sys
import json
import base64
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import transforms

# ===== Model Definition =====
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ===== Load Model Once =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=4).to(device)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "best_model.pth"))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform_inference = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
class_names = ["Resting", "Surveilling", "Activated", "Resolution"]

# ===== Analysis Function =====
def run_analysis(image_bytes: bytes, yolo_text: str):
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    width, height = original_image.size

    summary = {name: 0 for name in class_names}

    for idx, line in enumerate(yolo_text.strip().splitlines()):
        try:
            parts = list(map(float, line.strip().split()))
            if len(parts) not in [4, 5]:
                continue
            if len(parts) == 4:
                x_center, y_center, box_w, box_h = parts
            else:
                _, x_center, y_center, box_w, box_h = parts

            x = int((x_center - box_w / 2) * width)
            y = int((y_center - box_h / 2) * height)
            w = int(box_w * width)
            h = int(box_h * height)

            crop = annotated_image.crop((x, y, x + w, y + h))
            input_tensor = transform_inference(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, pred_class = torch.max(output, 1)

            class_name = class_names[pred_class.item()]
            summary[class_name] += 1

            draw.rectangle((x, y, x + w, y + h), outline="red", width=2)
            draw.text((x + 2, y - 10), class_name, fill="red")

        except Exception as e:
            print(f"❌ Error in cell {idx}: {e}", file=sys.stderr)
            continue

    # Convert images to base64
    annotated_buf = io.BytesIO()
    annotated_image.save(annotated_buf, format="PNG")
    annotated_b64 = base64.b64encode(annotated_buf.getvalue()).decode("utf-8")

    original_buf = io.BytesIO()
    original_image.save(original_buf, format="PNG")
    original_b64 = base64.b64encode(original_buf.getvalue()).decode("utf-8")

    return {
        "annotated_image_base64": annotated_b64,
        "converted_original_base64": original_b64,
        "summary": summary
    }

# ===== Optional: CLI Usage for Debugging =====
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze.py <image_path> <yolo_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    yolo_path = sys.argv[2]

    with open(image_path, "rb") as img_f, open(yolo_path, "r") as yolo_f:
        image_bytes = img_f.read()
        yolo_text = yolo_f.read()

    result = run_analysis(image_bytes, yolo_text)
    print(json.dumps(result))
