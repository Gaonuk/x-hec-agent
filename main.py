import os
from pydantic import BaseModel
from openai import OpenAI  # type: ignore
from fastapi import FastAPI, Depends, HTTPException
import replicate
from dotenv import load_dotenv  # type: ignore
import phospho  # type: ignore
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import FileResponse

load_dotenv()

client = OpenAI()
app = FastAPI()


phospho.init(
    api_key=os.getenv("PHOSPHO_API_KEY"), project_id=os.getenv("PHOSPHO_PROJECT_ID")
)

# input_str = "Hello! This is what the user asked to the system"
# output_str = "This is the response showed to the user by the app."
# phospho.log(input=input_str, output=output_str)

### (optional) Secure our API routes with an API key
bearer = HTTPBearer()


def get_api_hey(authorization: HTTPAuthorizationCredentials = Depends(bearer)) -> None:
    # Parse credentials
    api_key_token = authorization.credentials

    if api_key_token != os.getenv("MY_SECRET_KEY"):
        raise HTTPException(status_code=401, detail="Invalid token")


###


# Message model
class Message(BaseModel):
    message: str


@app.get("/")
def read_root():
    return {"Hello": "PLB!"}


@app.post("/message")
def send_message(request: Message):
    # Call OpenAI
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are Bruno Martinaud. Talk about your experience with OpenAI.",
            },
            {"role": "user", "content": request.message},
        ],
    )
    # Extract the message from the completion
    answer = completion.choices[0].message.content

    # Log to analytics
    phospho.log(input=request.message, output=answer)

    return {"answer": answer}


@app.post("/message-secure")
def send_message_secure(request: Message, api_key: str = Depends(get_api_hey)):
    # Call OpenAI
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are Bruno Martinaud. Talk about your experience with OpenAI.",
            },
            {"role": "user", "content": request.message},
        ],
    )
    # Extract the message from the completion
    answer = completion.choices[0].message.content

    return {"answer": answer}


@app.post("/image")
def send_image(request: Message):
    prompt = request.message

    # Call Replicate
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={
            "prompt": prompt,
            "go_fast": True,
            "megapixels": "1",
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "webp",
            "output_quality": 80,
            "num_inference_steps": 4,
        },
    )

    # Save the generated image
    with open("generated_image.webp", "wb") as f:
        f.write(output[0].read())

    # Return the image file
    return FileResponse(
        path="generated_image.webp",
        media_type="image/webp",
        filename="generated_image.webp",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
