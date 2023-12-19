import uvicorn
from fastapi import FastAPI, Query

from vision import api_text_recognition, api_classification

app = FastAPI(title='Timonin Nikita Yandex Cloud Vision Api')


@app.get("/text_recognition")
async def text_recognition(
        url_image: str = Query('https://yastat.net/s3/cloud/www/static/freeze/en/assets/img/vision.0bf4996d.png',
                               description='get text from image')
) -> dict:
    data = api_text_recognition(url_image)
    return data


@app.get("/classification")
async def classification(
        url_image: str = Query('https://yastat.net/s3/cloud/www/static/freeze/en/assets/img/vision.0bf4996d.png',
                               description='image moderation, face detection')
) -> dict:
    data = api_classification(url_image)
    return data


if __name__ == "__main__":
    uvicorn.run('api:app', host='127.0.0.1', port=8000, reload=True)
