import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
import os
from werkzeug.utils import secure_filename
import rembg


UPLOAD_FOLDER = 'static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.config = {
    'UPLOAD_FOLDER': UPLOAD_FOLDER,
    'DOWNLOAD_FOLDER': DOWNLOAD_FOLDER,
    'MAX_CONTENT_LENGTH': 10 * 1024 * 1024
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_folders():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)


def process_file(path, filename):
    bg_removed_path = os.path.join(app.config['UPLOAD_FOLDER'], "bg_removed_" + filename)
    remove_background(path, bg_removed_path)
    
    for i in range(5):
        modified_path = os.path.join(app.config['UPLOAD_FOLDER'], f"modified_{i+1}_" + filename)
        modify_image(path, modified_path, i)
        cartoonize(modified_path, filename, i)


def remove_background(path, output_path):
    with open(path, "rb") as input_file, open(output_path, "wb") as output_file:
        img_data = input_file.read()
        result = rembg.remove(img_data)
        output_file.write(result)


def modify_image(path, modified_path, index):
    image = Image.open(path)
    
    if index == 0:
        modified_image = image.filter(ImageFilter.BLUR)
    elif index == 1:
        modified_image = image.filter(ImageFilter.EMBOSS)
    elif index == 2:
        modified_image = image.rotate(45)
    elif index == 3:
        modified_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    elif index == 4:
        enhancer = ImageEnhance.Brightness(image)
        modified_image = enhancer.enhance(1.5)
    
    modified_image.save(modified_path)



def cartoonize(path, filename, index):
    weight = torch.load('weight.pth', map_location='cpu')
    model = SimpleGenerator()
    model.load_state_dict(weight)
    model.eval()
    image = Image.open(path)
    new_image = image.resize((256, 256))
    new_image.save(path)
    raw_image = cv2.imread(path)
    image = raw_image / 127.5 - 1
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image).unsqueeze(0)
    output = model(image.float())
    output = output.squeeze(0).detach().numpy()
    output = output.transpose(1, 2, 0)
    output = (output + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)

    cartoon_folder = os.path.join(app.config['DOWNLOAD_FOLDER'], 'cartoon')
    os.makedirs(cartoon_folder, exist_ok=True)
    output_path = os.path.join(cartoon_folder, f"cartoon_{index+1}_" + filename)

    cv2.imwrite(output_path, output)

    # Remove the background from the cartoon image
    bg_removed_output_path = os.path.join(cartoon_folder, f"bg_removed_cartoon_{index+1}_" + filename)
    remove_background(output_path, bg_removed_output_path)


@app.on_event("startup")
async def startup_event():
    create_folders()


@app.get("/")
async def index():
    return "Hello, please use the /toonify endpoint to upload an image."


@app.post("/toonify")
async def toonify(file: UploadFile = File(...)):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        process_file(file_path, filename)
        return {
            f"file{i+1}": FileResponse(
                os.path.join(app.config['DOWNLOAD_FOLDER'], 'cartoon', f"cartoon_{i+1}_" + filename),
                filename=f"cartoon_{i+1}_" + filename
            ) for i in range(5)
        }
    return {"error": "Invalid file format. Please upload an image file (PNG, JPG, JPEG)."}


class ResBlock(nn.Module):
    def __init__(self, num_channel):
        super(ResBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),
            nn.BatchNorm2d(num_channel))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        output = self.conv_layer(inputs)
        output = self.activation(output + inputs)
        return output


class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 2, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, inputs):
        output = self.conv_layer(inputs)
        return output


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_last=False):
        super(UpBlock, self).__init__()
        self.is_last = is_last
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1))
        self.act = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self.last_act = nn.Tanh()

    def forward(self, inputs):
        output = self.conv_layer(inputs)
        if self.is_last:
            output = self.last_act(output)
        else:
            output = self.act(output)
        return output


class SimpleGenerator(nn.Module):
    def __init__(self, num_channel=32, num_blocks=4):
        super(SimpleGenerator, self).__init__()
        self.down1 = DownBlock(3, num_channel)
        self.down2 = DownBlock(num_channel, num_channel * 2)
        self.down3 = DownBlock(num_channel * 2, num_channel * 3)
        self.down4 = DownBlock(num_channel * 3, num_channel * 4)
        res_blocks = [ResBlock(num_channel * 4)] * num_blocks
        self.res_blocks = nn.Sequential(*res_blocks)
        self.up1 = UpBlock(num_channel * 4, num_channel * 3)
        self.up2 = UpBlock(num_channel * 3, num_channel * 2)
        self.up3 = UpBlock(num_channel * 2, num_channel)
        self.up4 = UpBlock(num_channel, 3, is_last=True)

    def forward(self, inputs):
        d1 = self.down1(inputs)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        res = self.res_blocks(d4)
        u1 = self.up1(res)
        u2 = self.up2(u1 + d3)
        u3 = self.up3(u2 + d2)
        output = self.up4(u3 + d1)
        return output


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=12000)
