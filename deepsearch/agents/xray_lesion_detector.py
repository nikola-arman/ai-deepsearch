import ultralytics
from functools import lru_cache
import gdown
import os
import numpy as np
import cv2
from typing import Optional
from openai import OpenAI
import base64
import io
from PIL import Image

@lru_cache(maxsize=1)
def _load_model():
    tmp_path = os.path.join(os.getcwd(), 'cache', 'yolo11l-chess-xray-lesion-detector.pt')

    if not os.path.exists(tmp_path):
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        gdown.download(id='1GhdSBS8sS6tjX9-JnA6q9MIUOP9g3EQj', output=tmp_path)

    return ultralytics.YOLO(tmp_path)

@lru_cache(maxsize=1)
def _load_another_model():
    tmp_path = os.path.join(os.getcwd(), 'cache', 'yolo11l-chess-xray-lesion-detector-2.pt')

    if not os.path.exists(tmp_path):
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        gdown.download(id='1fiyUNbAuDCOBRt6cuiZynjOU4y5ELmbY', output=tmp_path)

    return ultralytics.YOLO(tmp_path)

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
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Nodule/Mass',
    'Other lesion',
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

def predict(image_path: str) -> PredictionResult:
    model = _load_model()
    another_model = _load_another_model()
    
    results = model(image_path, verbose=False)[0].boxes
    another_results = another_model(image_path, verbose=False)[0].boxes
    
    size = results.orig_shape

    return PredictionResult(
        xyxyn=results.xyxyn.cpu().numpy().tolist() + another_results.xyxyn.cpu().numpy().tolist(),
        conf=results.conf.cpu().numpy().tolist() + another_results.conf.cpu().numpy().tolist(),
        cls=results.cls.cpu().numpy().astype(int).tolist() + (another_results.cls.cpu().numpy() + 7).astype(int).tolist(),
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
    image = cv2.imread(result.org_path)
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


def xray_dianose_agent(img_path: str, orig_user_message: Optional[str] = None) -> tuple[Optional[np.ndarray], Optional[str]]:
    result = predict(img_path)
    res = quick_diagnose(result) 

    if res is not None:
        vis = visualize(result)
    else:
        vis = None

    if not res:
        client = OpenAI(
            base_url=os.getenv('VLM_BASE_URL'),
            api_key=os.getenv('VLM_API_KEY')
        )

        system_prompt = 'you are reading a user provided image (un categorized) and you need to provide a short comment on the image and give a diagnosis for the patient, if it is not medical or healthcare related docluemnts, just need to answer "looking good!" . Write the diagnosis in under 3 sentences, plain text only and no new lines.'

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        b = io.BytesIO()
        img.save(b, format='JPEG')
        image_uri = f'data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode("utf-8")}'

        comment_by_doctor = client.chat.completions.create(
            model=os.getenv('LLM_MODEL_ID'),
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
                            'text': orig_user_message or 'Read and analyze it! note for the important information. The image is a user provided; if nothing weird, just answer "looking good!".'
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
            max_tokens=512,
            temperature=0.2
        )

        return None, comment_by_doctor.choices[0].message.content

    else:
        client = OpenAI(
            base_url=os.getenv('LLM_BASE_URL'),
            api_key=os.getenv('LLM_API_KEY')
        )
            
        system_prompt = 'You are a doctor, you are reading an X-ray image of a patient and it has some lesions on it. You need to provide a short comment on the image and give a diagnosis for the patient. Notice that the Other lesion can be anything, but mostly backbone-related or bone breakage. Write the diagnosis in under 3 sentences, plain text only, no new lines.'
        
        comment_by_doctor = client.chat.completions.create(
            model=os.getenv('LLM_MODEL_ID'),
            messages=[
                {
                    'role': 'system', 
                    'content': system_prompt
                },
                {
                    'role': 'user', 
                    'content': f'Yolo v11l output: {res}'
                }
            ],
            max_tokens=256,
            temperature=0.2
        )

        return vis, comment_by_doctor.choices[0].message.content