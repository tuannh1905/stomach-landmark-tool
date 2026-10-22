
from pathlib import Path
import hashlib

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from models import create_model


# =========================================================
# 1. Page configuration
# =========================================================
st.set_page_config(
    page_title="Gastric Landmark AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 2. Compact, responsive styling
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1600px;
        padding: 0.6rem 1.4rem 0.3rem;
    }

    header[data-testid="stHeader"] {
        height: 2rem;
    }

    h1 {
        font-size: clamp(1.25rem, 2vw, 1.8rem) !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow-wrap: anywhere;
    }

    h2, h3 {
        font-size: 1rem !important;
        line-height: 1.2 !important;
        margin: 0.25rem 0 0.45rem !important;
        padding: 0 !important;
    }

    p {
        margin-bottom: 0.25rem;
    }

    [data-testid="stCaptionContainer"] {
        font-size: 0.78rem;
    }

    [data-testid="stFileUploader"] {
        margin: 0 !important;
    }

    [data-testid="stFileUploader"] section {
        padding: 0.35rem 0.5rem;
        min-height: 4rem;
    }

    div.stButton > button {
        width: 100%;
        min-height: 2.6rem;
        padding: 0.35rem;
        font-weight: 600;
    }

    [data-testid="stImage"] img {
        max-height: 43vh;
        width: 100%;
        object-fit: contain;
    }

    [data-testid="stMetric"] {
        padding: 0.3rem 0.65rem;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 0.5rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.65rem;
    }

    [data-testid="stAlert"] {
        padding: 0.55rem 0.75rem;
        margin: 0.3rem 0;
    }

    hr {
        margin: 0.35rem 0 !important;
    }

    .prob-row {
        margin: 0.4rem 0 0.55rem;
    }

    .prob-label {
        display: flex;
        justify-content: space-between;
        gap: 0.5rem;
        font-size: 0.83rem;
        margin-bottom: 0.2rem;
    }

    .prob-track {
        width: 100%;
        height: 0.42rem;
        border-radius: 99px;
        background: rgba(128,128,128,0.18);
        overflow: hidden;
    }

    .prob-fill {
        height: 100%;
        border-radius: 99px;
        background: #2784e8;
    }

    .footnote {
        font-size: 0.72rem;
        opacity: 0.7;
        margin-top: 0.3rem;
    }

    footer {
        display: none;
    }

    @media (max-height: 800px) {
        .block-container {
            padding-top: 0.25rem;
        }

        [data-testid="stImage"] img {
            max-height: 36vh;
        }

        .prob-row {
            margin: 0.25rem 0 0.35rem;
        }
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 0.7rem;
            padding-right: 0.7rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. Model configuration
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "best.pth"

CLASS_NAMES = [
    "Cardia",
    "Fundus",
    "Greater curvature",
    "Angulus",
    "Antrum",
]

CLASS_NAMES_VI = [
    "Tâm vị",
    "Đáy vị",
    "Bờ cong lớn",
    "Góc bờ cong nhỏ",
    "Hang vị",
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================================================
# 4. Load model once
# =========================================================
@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model_init = checkpoint["model_init"]

    model = create_model(
        num_classes=model_init["num_classes"],
        drop_path_rate=model_init["drop_path_rate"],
    )

    state_dict = checkpoint.get(
        "ema_state_dict",
        checkpoint["model_state_dict"],
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


# =========================================================
# 5. Header
# =========================================================
header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown(
        "<h1>🔬 Gastric Landmark AI</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Endoscopic image classification · 5 anatomical regions"
    )

with header_right:
    st.markdown(
        f"<div style='text-align:right; padding-top:0.4rem; "
        f"font-size:0.8rem;'>Device: <b>{device.type.upper()}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()


# =========================================================
# 6. Upload and classify controls
# =========================================================
upload_col, button_col = st.columns([3.2, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload an endoscopic image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        label_visibility="visible",
    )

with button_col:
    st.markdown(
        "<div style='height:1.7rem'></div>",
        unsafe_allow_html=True,
    )
    predict_clicked = st.button(
        "🔍 Classify Image",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

st.caption("Supported formats: JPG, PNG, BMP and TIFF · Model input: 224 × 224 pixels")
st.divider()


# =========================================================
# 7. Identify the currently uploaded image
# =========================================================
if "prediction" not in st.session_state:
    st.session_state.prediction = None

image = None
image_key = None

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image_key = hashlib.sha256(image_bytes).hexdigest()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Unable to read this image. Please try another file.")
        image = None

# Clear old results when the uploaded image changes
result = st.session_state.prediction

if result is not None and result["image_key"] != image_key:
    st.session_state.prediction = None
    result = None


# =========================================================
# 8. Main dashboard
# =========================================================
image_col, result_col = st.columns([1.05, 1])

with image_col:
    st.subheader("Input Image")

    if image is None:
        st.info("Upload an image above to get started.")
    else:
        st.image(image, use_container_width=True)
        st.caption(
            f"{image.width} × {image.height} pixels"
        )

with result_col:
    st.subheader("Prediction Results")

    if predict_clicked and image is not None:
        try:
            with st.spinner("Analyzing image..."):
                model = load_model()
                image_tensor = (
                    transform(image).unsqueeze(0).to(device)
                )

                with torch.inference_mode():
                    logits = model(image_tensor)
                    probabilities = torch.softmax(
                        logits, dim=1
                    )[0].cpu()

            predicted_index = probabilities.argmax().item()

            st.session_state.prediction = {
                "image_key": image_key,
                "index": predicted_index,
                "probabilities": probabilities.tolist(),
            }

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.session_state.prediction = None

    result = st.session_state.prediction

    if result is None:
        st.info(
            "The predicted region and class probabilities "
            "will appear here."
        )
    else:
        predicted_index = result["index"]
        probabilities = result["probabilities"]
        confidence = probabilities[predicted_index]

        st.success(
            f"**Predicted Region:** "
            f"{CLASS_NAMES[predicted_index]}"
        )

        st.metric(
            "Top-class probability",
            f"{confidence * 100:.2f}%",
        )

        st.markdown("**Class Probabilities**")

        for i, probability in enumerate(probabilities):
            percentage = probability * 100

            st.markdown(
                f"""
                <div class="prob-row">
                    <div class="prob-label">
                        <span>{CLASS_NAMES[i]}</span>
                        <b>{percentage:.1f}%</b>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill"
                             style="width:{percentage:.4f}%;">
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="footnote">'
            'Model probability is not equivalent to clinical accuracy.'
            '</div>',
            unsafe_allow_html=True,
        )


# =========================================================
# 9. Footer
# =========================================================
st.divider()

st.markdown(
    '<div class="footnote">'
    'Experimental research tool · Not intended for clinical '
    'diagnosis or treatment decisions.'
    '</div>',
    unsafe_allow_html=True,
)
