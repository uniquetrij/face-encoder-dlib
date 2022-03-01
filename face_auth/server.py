import uvicorn
from fastapi import FastAPI

from face_auth.models import RegRequest, AuthRequest, ImageRequest
from face_auth.utils import str_to_pil_image, pil_to_cv2_image, cv2_img_to_encodings
from face_auth.face_rec import build_clf, predict

app = FastAPI()


@app.post("/encode/")
async def do_auth(auth_req: ImageRequest):
    image = pil_to_cv2_image(str_to_pil_image(auth_req.image))
    encoding = cv2_img_to_encodings(image)
    return encoding.tolist()

@app.get("/")
async def root():
    return {"message": "OK"}


if __name__ == "__main__":
    build_clf()
    uvicorn.run(app, host="0.0.0.0", port=9988)
