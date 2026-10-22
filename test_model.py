
import torch
from models import create_model

# 1. Chọn thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 2. Đọc checkpoint
checkpoint = torch.load(
    "best.pth",
    map_location="cpu",
    weights_only=False
)

# 3. Khởi tạo kiến trúc đúng với checkpoint
model = create_model(
    num_classes=checkpoint["model_init"]["num_classes"],
    drop_path_rate=checkpoint["model_init"]["drop_path_rate"]
)

# 4. Nạp trọng số EMA
state_dict = checkpoint["ema_state_dict"]
model.load_state_dict(state_dict, strict=True)

# 5. Chuyển sang chế độ suy luận
model = model.to(device)
model.eval()

# 6. Tạo một ảnh giả có kích thước 224 x 224
x = torch.randn(1, 3, 224, 224).to(device)

# 7. Chạy dự đoán
with torch.inference_mode():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)

print("Logits:", logits)
print("Probabilities:", probabilities)
print("Predicted class:", probabilities.argmax(dim=1).item())
