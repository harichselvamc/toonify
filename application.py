import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import rembg
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
from werkzeug.utils import secure_filename
from torch import nn

# # app = FastAPI()

# # # Configure upload and download folders
# # UPLOAD_FOLDER = "./uploads"
# # DOWNLOAD_FOLDER = "./downloads"
# # ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# # shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
# # shutil.rmtree(DOWNLOAD_FOLDER, ignore_errors=True)
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


# # class Item(BaseModel):
# #     name: str
# #     description: str


# # def allowed_file(filename):
# #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# # def process_file(path, filename):
# #     bg_removed_path = os.path.join(UPLOAD_FOLDER, "bg_removed_" + filename)
# #     bg_removal(path, bg_removed_path)
# #     cartoonize(bg_removed_path, filename)


# # def bg_removal(path, output_path):
# #     with open(path, "rb") as input_file, open(output_path, "wb") as output_file:
# #         img_data = input_file.read()
# #         result = rembg.remove(img_data)
# #         output_file.write(result)


# # def cartoonize(path, filename):
# #     weight = torch.load('weight.pth', map_location='cpu')
# #     model = SimpleGenerator()
# #     model.load_state_dict(weight)
# #     model.eval()
# #     image = Image.open(path)
# #     new_image = image.resize((256, 256))
# #     new_image.save(path)
# #     raw_image = cv2.imread(path)
# #     image = raw_image / 127.5 - 1
# #     image = image.transpose(2, 0, 1)
# #     image = torch.tensor(image).unsqueeze(0)
# #     output = model(image.float())
# #     output = output.squeeze(0).detach().numpy()
# #     output = output.transpose(1, 2, 0)
# #     output = (output + 1) * 127.5
# #     output = np.clip(output, 0, 255).astype(np.uint8)

# #     cartoon_folder = os.path.join(DOWNLOAD_FOLDER, 'cartoon')
# #     os.makedirs(cartoon_folder, exist_ok=True)
# #     output_path = os.path.join(cartoon_folder, filename)

# #     cv2.imwrite(output_path, output)


# # @app.on_event("startup")
# # async def startup_event():
# #     pass


# # @app.get("/")
# # async def index():
# #     return "Hello, please use the /toonify endpoint to upload an image."


# # @app.post("/toonify")
# # async def toonify(file: UploadFile = File(...)):
# #     if file and allowed_file(file.filename):
# #         filename = secure_filename(file.filename)
# #         file_path = os.path.join(UPLOAD_FOLDER, filename)
# #         with open(file_path, "wb") as buffer:
# #             shutil.copyfileobj(file.file, buffer)
# #         process_file(file_path, filename)
# #         return FileResponse(os.path.join(DOWNLOAD_FOLDER, 'cartoon', filename), filename=filename)
# #     return {"error": "Invalid file format. Please upload an image file (PNG, JPG, JPEG)."}


# # class ResBlock(nn.Module):
# #     def __init__(self, num_channel):
# #         super(ResBlock, self).__init__()
# #         self.conv_layer = nn.Sequential(
# #             nn.Conv2d(num_channel, num_channel, 3, 1, 1),
# #             nn.BatchNorm2d(num_channel),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(num_channel, num_channel, 3, 1, 1),
# #             nn.BatchNorm2d(num_channel))
# #         self.activation = nn.ReLU(inplace=True)

# #     def forward(self, inputs):
# #         output = self.conv_layer(inputs)
# #         output = self.activation(output + inputs)
# #         return output


# # class DownBlock(nn.Module):
# #     def __init__(self, in_channel, out_channel):
# #         super(DownBlock, self).__init__()
# #         self.conv_layer = nn.Sequential(
# #             nn.Conv2d(in_channel, out_channel, 3, 2, 1),
# #             nn.BatchNorm2d(out_channel),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(out_channel, out_channel, 3, 1, 1),
# #             nn.BatchNorm2d(out_channel),
# #             nn.ReLU(inplace=True))

# #     def forward(self, inputs):
# #         output = self.conv_layer(inputs)
# #         return output


# # class UpBlock(nn.Module):
# #     def __init__(self, in_channel, out_channel, is_last=False):
# #         super(UpBlock, self).__init__()
# #         self.is_last = is_last
# #         self.conv_layer = nn.Sequential(
# #             nn.Conv2d(in_channel, in_channel, 3, 1, 1),
# #             nn.BatchNorm2d(in_channel),
# #             nn.ReLU(inplace=True),
# #             nn.Upsample(scale_factor=2),
# #             nn.Conv2d(in_channel, out_channel, 3, 1, 1))
# #         self.act = nn.Sequential(
# #             nn.BatchNorm2d(out_channel),
# #             nn.ReLU(inplace=True))
# #         self.last_act = nn.Tanh()

# #     def forward(self, inputs):
# #         output = self.conv_layer(inputs)
# #         if self.is_last:
# #             output = self.last_act(output)
# #         else:
# #             output = self.act(output)
# #         return output


# # class SimpleGenerator(nn.Module):
# #     def __init__(self, num_channel=32, num_blocks=4):
# #         super(SimpleGenerator, self).__init__()
# #         self.down1 = DownBlock(3, num_channel)
# #         self.down2 = DownBlock(num_channel, num_channel * 2)
# #         self.down3 = DownBlock(num_channel * 2, num_channel * 3)
# #         self.down4 = DownBlock(num_channel * 3, num_channel * 4)
# #         res_blocks = [ResBlock(num_channel * 4) for _ in range(num_blocks)]
# #         self.res_blocks = nn.Sequential(*res_blocks)
# #         self.up1 = UpBlock(num_channel * 4, num_channel * 3)
# #         self.up2 = UpBlock(num_channel * 3, num_channel * 2)
# #         self.up3 = UpBlock(num_channel * 2, num_channel)
# #         self.up4 = UpBlock(num_channel, 3, is_last=True)

# #     def forward(self, inputs):
# #         down1 = self.down1(inputs)
# #         down2 = self.down2(down1)
# #         down3 = self.down3(down2)
# #         down4 = self.down4(down3)
# #         down4 = self.res_blocks(down4)
# #         up1 = self.up1(down4)
# #         up2 = self.up2(up1 + down3)
# #         up3 = self.up3(up2 + down2)
# #         up4 = self.up4(up3 + down1)
# #         return up4


# # if __name__ == '__main__':
# #     import uvicorn

# #     uvicorn.run(app, host="0.0.0.0", port=12000)
import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import rembg
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
from werkzeug.utils import secure_filename
from torch import nn
from typing import List


# app = FastAPI()

# # Configure upload and download folders
# UPLOAD_FOLDER = "./uploads"
# DOWNLOAD_FOLDER = "./downloads"
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
# shutil.rmtree(DOWNLOAD_FOLDER, ignore_errors=True)
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


# class Item(BaseModel):
#     name: str
#     description: str


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def process_file(path, filename):
#     bg_removed_path = os.path.join(UPLOAD_FOLDER, "bg_removed_" + filename)
#     bg_removal(path, bg_removed_path)
#     cartoonize(bg_removed_path, filename)


# def bg_removal(path, output_path):
#     with open(path, "rb") as input_file, open(output_path, "wb") as output_file:
#         img_data = input_file.read()
#         result = rembg.remove(img_data)
#         output_file.write(result)


# def cartoonize(path, filename):
#     weight = torch.load('weight.pth', map_location='cpu')
#     model = SimpleGenerator()
#     model.load_state_dict(weight)
#     model.eval()
    
#     # Open and resize the image
#     image = Image.open(path)
#     new_image = image.resize((256, 256))
#     new_image.save(path)
    
#     # Perform cartoonization
#     raw_image = cv2.imread(path)
#     image = raw_image / 127.5 - 1
#     image = image.transpose(2, 0, 1)
#     image = torch.tensor(image).unsqueeze(0)
#     output = model(image.float())
#     output = output.squeeze(0).detach().numpy()
#     output = output.transpose(1, 2, 0)
#     output = (output + 1) * 127.5
#     output = np.clip(output, 0, 255).astype(np.uint8)
    
#     # Save the cartoon image
#     cartoon_folder = os.path.join(DOWNLOAD_FOLDER, 'cartoon')
#     os.makedirs(cartoon_folder, exist_ok=True)
#     output_path = os.path.join(cartoon_folder, filename)
#     cv2.imwrite(output_path, output)
    
#     # Make the background transparent
#     make_transparent(output_path)


# def make_transparent(image_path):
#     img = Image.open(image_path)
#     img = img.convert("RGBA")

#     datas = img.getdata()

#     new_data = []
#     for item in datas:
#         if item[0] == 255 and item[1] == 255 and item[2] == 255:
#             new_data.append((255, 255, 255, 0))
#         else:
#             new_data.append(item)

#     img.putdata(new_data)
#     img.save(image_path, "PNG")


# @app.on_event("startup")
# async def startup_event():
#     pass


# @app.get("/")
# async def index():
#     return "Hello, please use the /toonify endpoint to upload an image."


# @app.post("/toonify")
# async def toonify(file: UploadFile = File(...)):
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         process_file(file_path, filename)
#         return FileResponse(os.path.join(DOWNLOAD_FOLDER, 'cartoon', filename), filename=filename)
#     return {"error": "Invalid file format. Please upload an image file (PNG, JPG, JPEG)."}


# class ResBlock(nn.Module):
#     def __init__(self, num_channel):
#         super(ResBlock, self).__init__()
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(num_channel, num_channel, 3, 1, 1),
#             nn.BatchNorm2d(num_channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_channel, num_channel, 3, 1, 1),
#             nn.BatchNorm2d(num_channel))
#         self.activation = nn.ReLU(inplace=True)

#     def forward(self, inputs):
#         output = self.conv_layer(inputs)
#         output = self.activation(output + inputs)
#         return output


# class DownBlock(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(DownBlock, self).__init__()
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, 3, 2, 1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True))

#     def forward(self, inputs):
#         output = self.conv_layer(inputs)
#         return output


# class UpBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, is_last=False):
#         super(UpBlock, self).__init__()
#         self.is_last = is_last
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel, 3, 1, 1),
#             nn.BatchNorm2d(in_channel),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_channel, out_channel, 3, 1, 1))
#         self.act = nn.Sequential(
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True))
#         self.last_act = nn.Tanh()

#     def forward(self, inputs):
#         output = self.conv_layer(inputs)
#         if self.is_last:
#             output = self.last_act(output)
#         else:
#             output = self.act(output)
#         return output


# class SimpleGenerator(nn.Module):
#     def __init__(self, num_channel=32, num_blocks=4):
#         super(SimpleGenerator, self).__init__()
#         self.down1 = DownBlock(3, num_channel)
#         self.down2 = DownBlock(num_channel, num_channel * 2)
#         self.down3 = DownBlock(num_channel * 2, num_channel * 3)
#         self.down4 = DownBlock(num_channel * 3, num_channel * 4)
#         res_blocks = [ResBlock(num_channel * 4) for _ in range(num_blocks)]
#         self.res_blocks = nn.Sequential(*res_blocks)
#         self.up1 = UpBlock(num_channel * 4, num_channel * 3)
#         self.up2 = UpBlock(num_channel * 3, num_channel * 2)
#         self.up3 = UpBlock(num_channel * 2, num_channel)
#         self.up4 = UpBlock(num_channel, 3, is_last=True)

#     def forward(self, inputs):
#         down1 = self.down1(inputs)
#         down2 = self.down2(down1)
#         down3 = self.down3(down2)
#         down4 = self.down4(down3)
#         down4 = self.res_blocks(down4)
#         up1 = self.up1(down4)
#         up2 = self.up2(up1 + down3)
#         up3 = self.up3(up2 + down2)
#         up4 = self.up4(up3 + down1)
#         return up4


# if __name__ == '__main__':
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=12000)


import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import rembg
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
from werkzeug.utils import secure_filename
from torch import nn

app = FastAPI()

# Configure upload and download folders
UPLOAD_FOLDER = "./uploads"
DOWNLOAD_FOLDER = "./downloads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
shutil.rmtree(DOWNLOAD_FOLDER, ignore_errors=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)


class Item(BaseModel):
    name: str
    description: str


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_file(path, filename):
    bg_removed_path = os.path.join(UPLOAD_FOLDER, "bg_removed_" + filename)
    bg_removal(path, bg_removed_path)
    cartoonize(bg_removed_path, filename)


def bg_removal(path, output_path):
    with open(path, "rb") as input_file, open(output_path, "wb") as output_file:
        img_data = input_file.read()
        result = rembg.remove(img_data)
        output_file.write(result)


def cartoonize(path, filename):
    weight = torch.load('weight.pth', map_location='cpu')
    model = SimpleGenerator()
    model.load_state_dict(weight)
    model.eval()
    
    # Open and resize the image
    image = Image.open(path)
    new_image = image.resize((256, 256))
    new_image.save(path)
    
    # Perform cartoonization
    raw_image = cv2.imread(path)
    image = raw_image / 127.5 - 1
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image).unsqueeze(0)
    output = model(image.float())
    output = output.squeeze(0).detach().numpy()
    output = output.transpose(1, 2, 0)
    output = (output + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    # Save the cartoon image
    cartoon_folder = os.path.join(DOWNLOAD_FOLDER, 'cartoon')
    os.makedirs(cartoon_folder, exist_ok=True)
    output_path = os.path.join(cartoon_folder, filename)
    cv2.imwrite(output_path, output)
    
    # Make the background transparent
    make_transparent(output_path)


def make_transparent(image_path):
    img = Image.open(image_path)
    img = img.convert("RGBA")

    datas = img.getdata()

    new_data = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    img.putdata(new_data)
    img.save(image_path, "PNG")


@app.on_event("startup")
async def startup_event():
    pass


@app.get("/")
async def index():
    return "Hello, please use the /toonify endpoint to upload an image."


@app.post("/toonify")
async def toonify(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            process_file(file_path, filename)
            results.append(os.path.join(DOWNLOAD_FOLDER, 'cartoon', filename))
        else:
            return {"error": "Invalid file format. Please upload image files (PNG, JPG, JPEG)."}
    return {"results": results}


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
        res_blocks = [ResBlock(num_channel * 4) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        self.up1 = UpBlock(num_channel * 4, num_channel * 3)
        self.up2 = UpBlock(num_channel * 3, num_channel * 2)
        self.up3 = UpBlock(num_channel * 2, num_channel)
        self.up4 = UpBlock(num_channel, 3, is_last=True)

    def forward(self, inputs):
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down4 = self.res_blocks(down4)
        up1 = self.up1(down4)
        up2 = self.up2(up1 + down3)
        up3 = self.up3(up2 + down2)
        up4 = self.up4(up3 + down1)
        return up4


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=12000)
