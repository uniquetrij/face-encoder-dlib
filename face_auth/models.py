from pydantic.main import BaseModel

class ImageRequest(BaseModel):
    image: str

class RegRequest(BaseModel):
    image: str
    keyid: str


class AuthRequest(BaseModel):
    image: str
    therm: float
    sysid: str