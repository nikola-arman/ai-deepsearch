from functools import lru_cache
import gdown
import os
import numpy as np
import cv2
from typing import Optional, Tuple, List
from openai import OpenAI
import base64
import io
from PIL import Image
import onnxruntime as ort
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass


@lru_cache(maxsize=1)
def _load_model():
    tmp_path = os.path.join(os.getcwd(), 'cache', 'yolo11l-chess-xray-lesion-detector.onnx')

    if not os.path.exists(tmp_path):
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        gdown.download(id='1-4ogfKZsSnJMBEqUgk97WmhpPqnOMNVp', output=tmp_path)

    return ort.InferenceSession(tmp_path)

@lru_cache(maxsize=1)
def _load_another_model():
    tmp_path = os.path.join(os.getcwd(), 'cache', 'yolo11l-chess-xray-lesion-detector-2.onnx')

    if not os.path.exists(tmp_path):
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        gdown.download(id='1SDKL7x2RLy8FuBYucRyQq6YBERDIsXh9', output=tmp_path)

    return ort.InferenceSession(tmp_path)

def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize and reshape images while maintaining aspect ratio by adding padding.

    Args:
        img (np.ndarray): Input image to be resized.
        new_shape (Tuple[int, int]): Target shape (height, width) for the image.

    Returns:
        (np.ndarray): Resized and padded image.
        (Tuple[int, int]): Padding values (top, left) applied to the image.
    """
    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return img, (top, left)

@lru_cache(maxsize=1)
def preprocess(input_image: str) -> Tuple[np.ndarray, Tuple[int, int], int, int]:
    """
    Preprocess the input image before performing inference.

    This method reads the input image, converts its color space, applies letterboxing to maintain aspect ratio,
    normalizes pixel values, and prepares the image data for model input.

    Returns:
        (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
        (Tuple[int, int]): Padding values (top, left) applied during letterboxing.
    """
    # Read the input image using OpenCV
    img = Image.open(input_image)
    img = img.convert('RGB')
    img = np.array(img)

    # Get the height and width of the input image
    img_height, img_width = img.shape[:2]

    # Convert the image color space from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img, pad = letterbox(img, (640, 640))

    # Normalize the image data by dividing it by 255.0
    image_data = np.array(img) / 255.0

    # Transpose the image to have the channel dimension as the first dimension
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

    # Expand the dimensions of the image data to match the expected input shape
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

    # Return the preprocessed image data
    return image_data, pad, img_height, img_width

def postprocess(
    output: List[np.ndarray],
    img_height: int,
    img_width: int,
    pad: Tuple[int, int],
    confidence_thres: float = 0.2,
    iou_thres: float = 0.45
) -> np.ndarray:
    """
    Perform post-processing on the model's output to extract and visualize detections.

    This method processes the raw model output to extract bounding boxes, scores, and class IDs.
    It applies non-maximum suppression to filter overlapping detections and draws the results on the input image.

    Args:
        input_image (np.ndarray): The input image.
        output (List[np.ndarray]): The output arrays from the model.
        pad (Tuple[int, int]): Padding values (top, left) used during letterboxing.

    Returns:
        (np.ndarray): The input image with detections drawn on it.
    """
    # Transpose and squeeze the output to match the expected shape
    outputs = np.transpose(np.squeeze(output[0]))

    # Get the number of rows in the outputs array
    rows = outputs.shape[0]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Calculate the scaling factors for the bounding box coordinates
    gain = min(640 / img_height, 640 / img_width)
    outputs[:, 0] -= pad[1]
    outputs[:, 1] -= pad[0]

    # Iterate over each row in the outputs array
    for i in range(rows):
        # Extract the class scores from the current row
        classes_scores = outputs[i][4:]

        # Find the maximum score among the class scores
        max_score = np.amax(classes_scores)

        # If the maximum score is above the confidence threshold
        if max_score >= confidence_thres:
            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # Extract the bounding box coordinates from the current row
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Calculate the scaled coordinates of the bounding box
            left = int((x - w / 2) / gain)
            top = int((y - h / 2) / gain)
            width = int(w / gain)
            height = int(h / gain)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

    results = []

    # Iterate over the selected indices after non-maximum suppression
    for i in indices:
        # Get the box, score, and class ID corresponding to the index
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        results.append({
            "class_id": int(class_id),
            "score": float(score),
            "box": box
        })

    return results

from dataclasses import dataclass, field

CLS_NAMES = [
    "Disc Space Narrowing",
    "Foraminal Stenosis",
    "Osteophytes",
    "Other Lesion",
    "Spondylolysthesis",
    "Surgical Implant",
    "Vertebral Collapse",
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Consolidation',
    'Interstitial lung disease',
    'Infiltration',
    'Lung Opacity',
    'Nodule/Mass',
    'Other lesion', # keep duplications
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis'
]

ENABLED_CLS = [True] * len(CLS_NAMES)

@dataclass
class PredictionResult:
    xyxyn: list[list[float]] = field(default_factory=list)
    conf: list[float] = field(default_factory=list)
    cls: list[int] = field(default_factory=list)
    org_size: tuple[int, int] = field(default_factory=tuple) # height, width
    org_path: str = field(default_factory=str)

def infer(session: ort.InferenceSession, image_path: str) -> PredictionResult:
    image_data, pad, img_height, img_width = preprocess(image_path)
    outputs = session.run(None, {session.get_inputs()[0].name: image_data})
    outputs = postprocess(outputs, img_height, img_width, pad)

    for i in range(len(outputs)):
        x, y, w, h = outputs[i]['box']
        outputs[i]['xyxyn'] = [x / img_width, y / img_height, (x + w) / img_width, (y + h) / img_height]

    res = PredictionResult(
        xyxyn=[i['xyxyn'] for i in outputs],
        conf=[i['score'] for i in outputs],
        cls=[i['class_id'] for i in outputs],
        org_size=(img_height, img_width),
        org_path=image_path,
    )

    return res

def predict(image_path: str) -> PredictionResult:
    model = _load_model()
    another_model = _load_another_model()

    results = infer(model, image_path)
    another_results = infer(another_model, image_path)

    for i in range(len(another_results.cls)):
        another_results.cls[i] += 7

    size = results.org_size

    return PredictionResult(
        xyxyn=results.xyxyn + another_results.xyxyn,
        conf=results.conf + another_results.conf,
        cls=results.cls + another_results.cls,
        org_size=size,
        org_path=image_path,
    )

from enum import Enum

class RelativePosition(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"

    def __str__(self):
        return {
            "left": "left",
            "right": "right",
            "top": "top",
            "bottom": "bottom",
            "center": "center",
            "top_left": "upper left",
            "top_right": "upper right",
            "bottom_left": "lower left",
            "bottom_right": "lower right",
        }[self.value]

def calc_relative_position(xyxyn: list[list[float]]) -> RelativePosition:
    cny, cnx = (xyxyn[0] + xyxyn[2]) / 2, (xyxyn[1] + xyxyn[3]) / 2

    if 0.3 < cnx < 0.7 and 0.3 < cny < 0.7:
        return RelativePosition.CENTER

    if cnx < 0.3 and cny < 0.3:
        return RelativePosition.TOP_LEFT

    if cnx < 0.3 and cny > 0.7:
        return RelativePosition.BOTTOM_LEFT

    if cnx > 0.7 and cny > 0.7:
        return RelativePosition.BOTTOM_RIGHT

    if cnx > 0.7 and cny < 0.3:
        return RelativePosition.TOP_RIGHT

    if cnx < 0.3:
        return RelativePosition.LEFT

    if cnx > 0.7:
        return RelativePosition.RIGHT

    if cny < 0.3:
        return RelativePosition.TOP

    if cny > 0.7:
        return RelativePosition.BOTTOM

def quick_diagnose(result: PredictionResult) -> str:
    by_relative_position = {}
    total = 0

    for xyxyn, conf, cls in zip(result.xyxyn, result.conf, result.cls):
        if ENABLED_CLS[cls]:
            total += 1
            relative_position = calc_relative_position(xyxyn)

            if relative_position not in by_relative_position:
                by_relative_position[relative_position] = []

            by_relative_position[relative_position].append({
                "conf": conf,
                "lesion": CLS_NAMES[cls],
            })

    if total == 0:
        return None

    res = 'Found {} lesions in total:\n'.format(total)

    for relative_position, lesions in by_relative_position.items():
        res += '{}: {}'.format(
            str(relative_position),
            ', '.join(
                ['{} (conf: {:.2f})'.format(lesion['lesion'], lesion['conf']) for lesion in lesions]
            )
        )

    return res

def visualize(result: PredictionResult) -> np.ndarray:
    image = Image.open(result.org_path)
    logger.info("Visualizing image mode: {}".format(image.mode))
    image = image.convert('RGB')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    PAD = 5

    for xyxyn, conf, cls in zip(result.xyxyn, result.conf, result.cls):
        if ENABLED_CLS[cls]:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = xyxyn

            x1 = max(0, int(x1 * w) - PAD)
            y1 = max(0, int(y1 * h) - PAD)
            x2 = min(w, int(x2 * w) + PAD)
            y2 = min(h, int(y2 * h) + PAD)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.putText(image, CLS_NAMES[cls], (x1 + 5, y1 + 5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

def is_xray_image(img_path: str) -> bool:

    img = Image.open(img_path)
    # Convert to RGB
    rgb_img = img.convert('RGB')
    b = io.BytesIO()
    rgb_img.save(b, format='JPEG')

    image_uri = f'data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode("utf-8")}'
    system_prompt = 'you are classifying whether the image is a xray image or not. just answer "yes" or "no" in plain text.'

    client = OpenAI(
        base_url=os.getenv('LLM_BASE_URL'),
        api_key=os.getenv('LLM_API_KEY')
    )
    out = client.chat.completions.create(
        model=os.getenv('LLM_MODEL_ID', 'local-model'),
        messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            "url": image_uri
                        }
                    },
                    {
                        'type': 'text',
                        'text': 'is this a xray image?'
                    }
                ]
            }
        ],
        max_tokens=10,
        temperature=0.2
    )
    logger.info(f"Response from LLM: {out.choices[0].message.content}")
    msg_out = out.choices[0].message.content
    return 'yes' in msg_out.lower()


def xray_diagnose_agent(
    img_path: str,
    orig_user_message: Optional[str] = None,
) -> tuple[bool, Optional[np.ndarray], Optional[str]]:
    """Return the diagnosis of the image.

    Returns:
        - is_xray: bool, whether the image is xray or not
        - vis: np.ndarray, the image with bounding boxes (can be None)
        - comment_by_doctor: str, the comment by doctor
    """
    is_xray = is_xray_image(img_path)

    if not is_xray:
        client = OpenAI(
            base_url=os.getenv('LLM_BASE_URL'),
            api_key=os.getenv('LLM_API_KEY')
        )

        system_prompt = 'You are a healthcare master, you are reading and diagnosing an image for a user. Notice that the image can be a medical report, in-body, blood test report, skin, face, or other parts, and the problem can be lesions, fractures, etc. Keep the conversation concise. If it is medical or healthcare-related documents, or something like an in-body report, BMI report, prescription, etc, extract the content and summarize it. Otherwise, if the image is a body part, face, write a short medical diagnosis if something is wrong, or just answer "looking good!". Just  write diagnosis, no recommendation needed.'

        img = Image.open(img_path)
        rgb_img = img.convert('RGB')
        b = io.BytesIO()
        rgb_img.save(b, format='JPEG')
        image_uri = f'data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode("utf-8")}'

        comment_by_doctor = client.chat.completions.create(
            model=os.getenv('LLM_MODEL_ID', 'local-model'),
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': orig_user_message or 'Help me read and diagnose it!'
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                "url": image_uri
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024,
            temperature=0.2
        )

        return False, None, comment_by_doctor.choices[0].message.content

    result = predict(img_path)
    res = quick_diagnose(result)

    if res is not None:
        vis = visualize(result)
    else:
        vis = None

    client = OpenAI(
        base_url=os.getenv('LLM_BASE_URL'),
        api_key=os.getenv('LLM_API_KEY')
    )

    system_prompt = 'You are a doctor, you are reading an X-ray image of a patient and it has some lesions on it. You need to provide a short comment on the image and give a diagnosis for the patient. Notice that the Other lesion can be anything, but mostly backbone-related or bone breakage. Write the diagnosis in under 3 sentences, plain text only, no new lines.'

    comment_by_doctor = client.chat.completions.create(
        model=os.getenv('LLM_MODEL_ID', 'local-model'),
        messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': f'Yolo v11l output: {res or "no lesions found"}'
            }
        ],
        max_tokens=256,
        temperature=0.2
    )

    return True, vis, comment_by_doctor.choices[0].message.content
