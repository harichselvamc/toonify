import os
import shutil
from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import rembg
import numpy as np

# Create a folder to store uploaded files and downloaded images
UPLOAD_FOLDER = "uploads"
DOWNLOAD_FOLDER = "downloads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(DOWNLOAD_FOLDER, "cartoon"), exist_ok=True)
os.makedirs(os.path.join(DOWNLOAD_FOLDER, "merged"), exist_ok=True)

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained cartoonization model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load("openai/CLIP", "ViT-B/32", pretrained=True).to(device).eval()
cartoon_model = torch.hub.load("CompVis/taming-transformers", "taming-transformer-cpk").to(device).eval()
load_size = 256


class ImageData(BaseModel):
    filename: str
    url: str


def allowed_file(filename):
    allowed_extensions = ["png", "jpg", "jpeg"]
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def secure_filename(filename):
    # Remove special characters and spaces from filename
    return "".join(c if c.isalnum() else "_" for c in filename)


def process_file(file_path, filename):
    image = Image.open(file_path).convert("RGB")
    image = image.resize((load_size, load_size), Image.LANCZOS)

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Generate the cartoonized image
    with torch.no_grad():
        _, _, rendered_layers, _, _ = cartoon_model(image_tensor)

    # Convert the rendered layers to PIL images
    rendered_images = []
    for layer in rendered_layers:
        layer = layer.clamp(-1, 1).squeeze(0).permute(1, 2, 0)
        rendered_images.append(to_pil_image(layer))

    # Save the cartoonized image
    cartoon_image_path = os.path.join(DOWNLOAD_FOLDER, "cartoon", filename)
    rendered_images[0].save(cartoon_image_path, "PNG", optimize=True, compress_level=0)

    # Save the rendered layers as separate images
    for i, layer_image in enumerate(rendered_images):
        layer_path = os.path.join(DOWNLOAD_FOLDER, f"rendered_{i}", filename)
        layer_image.save(layer_path, "PNG", optimize=True, compress_level=0)


def merge_images(generated_image_path, text, output_image_path):
    # Load the generated image and resize it
    generated_image = cv2.imread(generated_image_path)
    generated_image = cv2.resize(generated_image, (load_size, load_size))

    # Remove the alpha channel if it exists
    if generated_image.shape[2] == 4:
        generated_image = generated_image[:, :, :3]

    # Remove the background using rembg library
    with rembg.Image(np.array(generated_image)) as img:
        img = img.resize(generated_image.shape[:2])
        img = np.array(img)

    # Convert the image to PIL format and apply the text overlay
    merged_image = Image.fromarray(img)
    draw = ImageDraw.Draw(merged_image)
    text_position = (20, 20)
    text_color = (255, 255, 255)
    font = ImageFont.truetype("arial.ttf", 40)
    draw.text(text_position, text, fill=text_color, font=font)

    # Save the merged image
    merged_image.save(output_image_path)


def generate_html(images: List[ImageData]):
    css = '''
    <style>
        .overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            pointer-events: none;
        }
        
        .overlay img {
            pointer-events: auto;
            max-width: 100%;
            max-height: 100%;
            opacity: 0.7;
        }
        
        .container {
            display: flex;
            flex-wrap: wrap;
        }
        
        .image-card {
            flex-basis: 50%;
            padding: 10px;
        }
        
        .image-card img {
            max-width: 100%;
        }
        
        .download-button {
            padding: 10px;
            margin-top: 10px;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            display: inline-block;
            border-radius: 4px;
            transition: background-color 0.3s ease;
        }
        
        .download-button:hover {
            background-color: #45a049;
        }
    </style>
    '''

    js = '''
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        $(document).ready(function() {
            $('.image-card').click(function() {
                $('.image-card').removeClass('selected');
                $(this).addClass('selected');
            });
            
            $('.image-card').draggable({
                containment: 'parent',
                scroll: false
            });
        });
    </script>
    '''

    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cartoonizer</title>
        {css}
    </head>
    <body>
        <h1>Cartoonizer</h1>
        <div class="container">
            <div class="image-card">
                <h3>Title Image Overlay</h3>
                <div class="overlay">
                    <img src="https://example.com/title_image.jpg" alt="Title Image Overlay">
                </div>
            </div>
            <div class="image-card">
                <h3>Image Preview Screen</h3>
                <div class="overlay">
                    <img src="https://example.com/preview_image.jpg" alt="Image Preview">
                </div>
            </div>
            <div class="image-card">
                <h3>Upload Image</h3>
                <form action="/upload" method="post" enctype="multipart/form-data">
                    <input type="file" name="image_file">
                    <button type="submit">Upload</button>
                </form>
            </div>
            <div class="image-card">
                <h3>Background Text</h3>
                <form action="/generate" method="post">
                    <input type="text" name="background_text" placeholder="Enter background text">
                    <button type="submit">Generate</button>
                </form>
            </div>
        </div>
        <h2>Generated Images</h2>
        <div class="container">
    '''

    for image in images:
        html += f'''
            <div class="image-card">
                <img src="{image.url}" alt="{image.filename}">
                <a class="download-button" href="{image.url}" download="{image.filename}">Download</a>
            </div>
        '''

    html += '''
        </div>
        {js}
    </body>
    </html>
    '''

    return html


@app.post("/upload")
async def upload_image(image_file: UploadFile = File(...)):
    if allowed_file(image_file.filename):
        # Save the uploaded image
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(image_file.file, f)

        # Process the uploaded image
        process_file(file_path, filename)

        # Generate the HTML response
        images = [
            ImageData(filename=filename, url=f"/{DOWNLOAD_FOLDER}/cartoon/{filename}"),
            ImageData(filename=filename, url=f"/{DOWNLOAD_FOLDER}/rendered_0/{filename}"),
            ImageData(filename=filename, url=f"/{DOWNLOAD_FOLDER}/rendered_1/{filename}"),
            ImageData(filename=filename, url=f"/{DOWNLOAD_FOLDER}/rendered_2/{filename}")
        ]
        html_content = generate_html(images)

        # Save the HTML page
        html_path = os.path.join(DOWNLOAD_FOLDER, "index.html")
        with open(html_path, "w") as f:
            f.write(html_content)

        return HTMLResponse(content=html_content, status_code=200)
    else:
        return {"error": "Invalid file format. Only PNG, JPG, and JPEG files are allowed."}


@app.post("/generate")
async def generate_image(background_text: str = Form(...)):
    # Merge the images with the given background text
    generated_images = []
    for file_name in os.listdir(os.path.join(DOWNLOAD_FOLDER, "cartoon")):
        generated_image_path = os.path.join(DOWNLOAD_FOLDER, "cartoon", file_name)
        output_image_path = os.path.join(DOWNLOAD_FOLDER, "merged", file_name)
        merge_images(generated_image_path, background_text, output_image_path)
        generated_images.append(ImageData(filename=file_name, url=f"/{DOWNLOAD_FOLDER}/merged/{file_name}"))

    # Generate the HTML response
    html_content = generate_html(generated_images)

    # Save the HTML page
    html_path = os.path.join(DOWNLOAD_FOLDER, "index.html")
    with open(html_path, "w") as f:
        f.write(html_content)

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/")
async def home():
    # Return the generated HTML page
    html_path = os.path.join(DOWNLOAD_FOLDER, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    else:
        return {"error": "HTML page not found. Upload an image to generate the page."}


@app.get("/{folder}/{filename}")
async def download_file(folder: str, filename: str):
    # Return the requested file for download
    file_path = os.path.join(DOWNLOAD_FOLDER, folder, filename)
    return FileResponse(file_path, filename=filename)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
