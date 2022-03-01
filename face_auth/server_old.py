import os
from errno import EEXIST
from threading import Thread

import cv2
import uvicorn
from fastapi import FastAPI

from face_auth.models import RegRequest, AuthRequest
from face_auth.utils import str_to_pil_image, pil_to_cv2_image, cv2_img_to_encodings
from face_auth.face_rec import build_clf, predict

app = FastAPI()


@app.post("/reg/")
async def do_reg(reg_req: RegRequest):
    image = pil_to_cv2_image(str_to_pil_image(reg_req.image))
    keyid = reg_req.keyid
    try:
        os.mkdir('face_auth/faces/' + keyid)
    except OSError as e:
        if e.errno != EEXIST:
            raise
    count = len([name for name in os.listdir('face_auth/faces/' + keyid + '/') if os.path.isfile(name)])
    cv2.imwrite('face_auth/faces/' + keyid + '/' + str(count + 1) + '.jpg', image)
    Thread(target=build_clf).start()
    return {}


@app.post("/auth/")
async def do_auth(auth_req: AuthRequest):
    image = pil_to_cv2_image(str_to_pil_image(auth_req.image))
    encoding = cv2_img_to_encodings(image)
    # keyid = CLASSIFIER.predict([encoding])[0]
    # score = dict(zip(CLASSIFIER.classes_, CLASSIFIER.predict_proba([encoding])[0])).get(keyid)
    keyid, score = predict(encoding)
    return {
        'identity': str(keyid),
        'confidence': str(score),
        'auth': 'success' if score > .925 else 'fail',
        'therm': 'pass' if auth_req.therm < 100 else 'fail'
    }

    # res, im_png = cv2.imencode(".png", image)
    # return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    build_clf()
    uvicorn.run(app, host="0.0.0.0", port=9988)
