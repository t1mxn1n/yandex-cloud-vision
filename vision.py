import base64
import json
import os
from io import BytesIO

import requests
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()

iam_token = os.getenv('iam_token')
catalog_id = os.getenv('catalog_id')
imgur_client_id = os.getenv('imgur_client_id')


def upload_image(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image


def pil_to_base64(pil_img):
    img_buffer = BytesIO()
    pil_img.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode("utf-8")


def body_text_recognition(b64_img):
    return json.dumps(
        {
            "mimeType": "JPEG",
            "languageCodes": ["*"],
            "model": "page",
            "content": b64_img,
        }
    )


def api_text_recognition(img_url):

    url = 'https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {iam_token}",
        "x-folder-id": f"{catalog_id}",
        "x-data-logging-enabled": "true"
    }
    b64_img = pil_to_base64(upload_image(img_url))
    response = requests.post(url, data=body_text_recognition(b64_img), headers=headers)
    if not response:
        return {'error': 'not response', 'code': response.status_code, 'msg': response.text}
    response_json = response.json()
    if response.status_code == 200:
        return {'text': response_json['result']['textAnnotation']['fullText']}
    return {'error': response.json()}


def body_classification(b64_img):
    return json.dumps(
        {
            "folderId": catalog_id,
            "analyze_specs": [{
                "content": b64_img,
                "features": [{
                    "type": "CLASSIFICATION",
                    "classificationConfig": {
                        "model": "moderation"
                    }
                }, {"type": "FACE_DETECTION"}]
            }]
        }
    )


def api_classification(img_url):
    url = 'https://vision.api.cloud.yandex.net/vision/v1/batchAnalyze'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {iam_token}"
    }
    pil_img = upload_image(img_url)
    b64_img = pil_to_base64(pil_img)
    response = requests.post(url, data=body_classification(b64_img), headers=headers)

    if not response:
        return {'error': 'not response', 'code': response.status_code, 'msg': response.text}

    response_json = response.json()
    faces = response_json['results'][0]['results'][1]['faceDetection']
    faces_url = draw_rectangles_faces(pil_img, faces['faces']) if faces else {}

    if response.status_code == 200:
        return {'properties': response_json['results'][0]['results'][0]['classification']['properties']} | faces | faces_url
    return {'error': response_json}


def draw_rectangles_faces(img, faces):
    draw = ImageDraw.Draw(img)
    for face in faces:
        lines = face['boundingBox']['vertices']
        rectangle = (int(lines[0]['x']), int(lines[0]['y']), int(lines[2]['x']), int(lines[2]['y']))
        draw.rectangle(rectangle, outline="red", width=4)
    return imgur_upload(img)


def imgur_upload(pil_img):
    url = "https://api.imgur.com/3/image"

    payload = {'image': pil_to_base64(pil_img),
               'description': 'face detection (lab work for cloud technology)'}

    headers = {
        'Authorization': f'Client-ID {imgur_client_id}'
    }

    response = requests.post(url, headers=headers, data=payload)
    if not response:
        return {'error': 'not response', 'code': response.status_code}
    response_json = response.json()
    if response.status_code == 200:
        return {'link': response_json['data']['link']}
    return {'error': response_json}


if __name__ == '__main__':
    print(api_text_recognition("https://yastat.net/s3/cloud/www/static/freeze/en/assets/img/vision.0bf4996d.png"))
