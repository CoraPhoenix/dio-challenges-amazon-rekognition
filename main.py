from pathlib import Path
import os
from dotenv import load_dotenv

import boto3
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

# initialising Amazon Rekognition service
client = boto3.client("rekognition", region_name = "us-east-1",
                          aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
                          aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"))

# returns image path
def get_path(file_name : str) -> str:
    return str(Path(__file__).parent / "images" / file_name)

def recognise_celebrities(photo: str):
    """Returns info regarding detected celebrities using Amazon Rekognition"""
    with open(photo, "rb") as img:
        return client.recognize_celebrities(Image = {"Bytes" : img.read()})

def draw_boxes(image: str, output_path: str, face_details: str):

    """
    Draws boxes around faces of detected celebrities in a given image
    """
    img = Image.open(image)

    if img.mode == 'RGBA':
        img = img.convert('RGB')

    draw = ImageDraw.Draw(img) # ImageDraw instance to draw a box
    font = ImageFont.truetype("arial.ttf", 20) # ImageFont instance to write on the image

    width, height = img.size # get image dims

    for face in face_details:

        # get coordinates to draw a box
        box = face["Face"]["BoundingBox"]
        left = int(box["Left"] * width)
        top = int(box["Top"] * height)
        right = int((box["Left"] + box["Width"]) * width)
        bottom = int((box["Top"] + box["Height"]) * height)

        confidence = face.get("MatchConfidence", 0) # get confidence score

        # if confidence if big enough, draw box around face and put celebrity's name on it
        if confidence > 90:
            draw.rectangle([left, top, right, bottom], outline = "green", width = 5)

            text = face.get("Name", "")
            text_pos = (left, top - 20)
            bbox = draw.textbbox(text_pos, text, font = font)
            draw.rectangle(bbox, fill = "green")
            draw.text(text_pos, text, font = font, fill = "white")

    # save image
    img.save(output_path)
    print(f"Image was saved in location: {output_path}")


if __name__ == "__main__":

    # get list of images in "images" folder
    photo_list = [photo for photo in os.listdir(str(Path(__file__).parent / "images")) if not "boxes" in photo]

    for photo in photo_list:
        response = recognise_celebrities(get_path(photo)) # get response from Amazon Rekognition
        faces = response["CelebrityFaces"] # get all detected celebrity faces

        if not faces:
            print(f"No celebrity faces were detected for file: {get_path(photo)}")
            continue

        output = get_path(f"{Path(photo).stem}_boxes.jpg")
        draw_boxes(get_path(photo), output, faces) # draw boxes around detected celebrity faces

