# import os
# import cv2
# import numpy as np
# from PIL import Image, ImageFilter, ImageOps, ImageEnhance
# from fastapi import FastAPI, UploadFile, File
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# import torch
# import torch.nn as nn
# from werkzeug.utils import secure_filename
# import rembg
# from typing import List


# UPLOAD_FOLDER = 'static/uploads/'
# DOWNLOAD_FOLDER = 'static/downloads/'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# #DIR_PATH = os.path.dirname(os.path.realpath(_file_))
# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
# app.config = {
#     'UPLOAD_FOLDER': UPLOAD_FOLDER,
#     'DOWNLOAD_FOLDER': DOWNLOAD_FOLDER,
#     'MAX_CONTENT_LENGTH': 10 * 1024 * 1024
# }


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def create_folders():
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)


# def process_file(path, filename):
#     bg_removed_path = os.path.join(app.config['UPLOAD_FOLDER'], "bg_removed_" + filename)
#     remove_background(path, bg_removed_path)
    
#     for i in range(5):
#         modified_path = os.path.join(app.config['UPLOAD_FOLDER'], f"modified_{i+1}_" + filename)
#         modify_image(path, modified_path, i)
#         cartoonize(modified_path, filename, i)


# def remove_background(path, output_path):
#     with open(path, "rb") as input_file, open(output_path, "wb") as output_file:
#         img_data = input_file.read()
#         result = rembg.remove(img_data)
#         output_file.write(result)


# def modify_image(path, modified_path, index):
#     image = Image.open(path)
    
#     if index == 0:
#         modified_image = image.filter(ImageFilter.BLUR)
#     elif index == 1:
#         modified_image = image.filter(ImageFilter.EMBOSS)
#     elif index == 2:
#         modified_image = image.rotate(45)
#     elif index == 3:
#         modified_image = image.transpose(Image.FLIP_TOP_BOTTOM)
#     elif index == 4:
#         enhancer = ImageEnhance.Brightness(image)
#         modified_image = enhancer.enhance(1.5)
    
#     modified_image.save(modified_path)


# def cartoonize(path, filename, index):
#     weight = torch.load('weight.pth', map_location='cpu')
#     model = SimpleGenerator()
#     model.load_state_dict(weight)
#     model.eval()
#     image = Image.open(path)
#     new_image = image.resize((256, 256))
#     new_image.save(path)
#     raw_image = cv2.imread(path)
#     image = raw_image / 127.5 - 1
#     image = image.transpose(2, 0, 1)
#     image = torch.tensor(image).unsqueeze(0)
#     output = model(image.float())
#     output = output.squeeze(0).detach().numpy()
#     output = output.transpose(1, 2, 0)
#     output = (output + 1) * 127.5
#     output = np.clip(output, 0, 255).astype(np.uint8)

#     cartoon_folder = os.path.join(app.config['DOWNLOAD_FOLDER'], 'cartoon')
#     os.makedirs(cartoon_folder, exist_ok=True)
#     output_path = os.path.join(cartoon_folder, f"cartoon_{index+1}_" + filename)

#     cv2.imwrite(output_path, output)

#     # Remove the background from the cartoon image
#     bg_removed_output_path = os.path.join(cartoon_folder, f"bg_removed_cartoon_{index+1}_" + filename)
#     remove_background(output_path, bg_removed_output_path)


# @app.on_event("startup")
# async def startup_event():
#     create_folders()


# @app.get("/")
# async def index():
#     return "Hello, please use the /toonify endpoint to upload an image."


# # @app.post("/toonify")
# # async def toonify(files: List[UploadFile] = File(...)):
# #     responses = []
# #     for file in files:
# #         if file and allowed_file(file.filename):
# #             filename = secure_filename(file.filename)
# #             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #             with open(file_path, "wb") as buffer:
# #                 buffer.write(await file.read())
# #             process_file(file_path, filename)
# #             for i in range(5):
# #                 responses.append(
# #                     FileResponse(
# #                         os.path.join(app.config['DOWNLOAD_FOLDER'], 'cartoon', f"cartoon_{i+1}_" + filename),
# #                         filename=f"cartoon_{i+1}_" + filename
# #                     )
# #                 )
# #         else:
# #             return {"error": "Invalid file format. Please upload an image file (PNG, JPG, JPEG)."}
# #     return responses
# # @app.post("/toonify")
# # async def toonify(files: List[UploadFile] = File(...)):
# #     responses = []
# #     for file in files:
# #         if file and allowed_file(file.filename):
# #             filename = secure_filename(file.filename)
# #             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #             with open(file_path, "wb") as buffer:
# #                 buffer.write(await file.read())
# #             process_file(file_path, filename)
# #             for i in range(5):
# #                 responses.append(
# #                     FileResponse(
# #                         os.path.join(app.config['DOWNLOAD_FOLDER'], 'cartoon', f"cartoon_{i+1}_" + filename),
# #                         filename=f"cartoon_{i+1}_" + filename
# #                     )
# #                 )
# #         else:
# #             return {"error": "Invalid file format. Please upload an image file (PNG, JPG, JPEG)."}
# #     return responses
# import zipfile

# def create_zip_file(directory, zip_filename):
#     zip_path = os.path.join(app.config['DOWNLOAD_FOLDER'], zip_filename)
#     with zipfile.ZipFile(zip_path, 'w') as zip_file:
#         for root, _, files in os.walk(directory):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 zip_file.write(file_path, os.path.relpath(file_path, directory))
#     return zip_path

# @app.post("/toonify")
# async def toonify(files: List[UploadFile] = File(...)):
#     responses = []
#     images_folder = os.path.join(app.config['DOWNLOAD_FOLDER'], 'cartoon')
#     os.makedirs(images_folder, exist_ok=True)

#     for file in files:
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             with open(file_path, "wb") as buffer:
#                 buffer.write(await file.read())
#             process_file(file_path, filename)
#             for i in range(5):
#                 responses.append(
#                     FileResponse(
#                         os.path.join(images_folder, f"cartoon_{i+1}_" + filename),
#                         filename=f"cartoon_{i+1}_" + filename
#                     )
#                 )

#     zip_filename = 'cartoon_images.zip'
#     zip_path = create_zip_file(images_folder, zip_filename)
#     responses.append(FileResponse(zip_path, filename=zip_filename))

#     return responses


# class ResBlock(nn.Module):
#     def _init_(self, num_channel):
#         super(ResBlock, self)._init_()
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
#     def _init_(self, in_channel, out_channel):
#         super(DownBlock, self)._init_()
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
#     def _init_(self, in_channel, out_channel, is_last=False):
#         super(UpBlock, self)._init_()
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
#     def _init_(self, num_channel=32, num_blocks=4):
#         super(SimpleGenerator, self)._init_()
#         self.down1 = DownBlock(3, num_channel)
#         self.down2 = DownBlock(num_channel, num_channel * 2)
#         self.down3 = DownBlock(num_channel * 2, num_channel * 3)
#         self.down4 = DownBlock(num_channel * 3, num_channel * 4)
#         res_blocks = [ResBlock(num_channel * 4)] * num_blocks
#         self.res_blocks = nn.Sequential(*res_blocks)
#         self.up1 = UpBlock(num_channel * 4, num_channel * 3)
#         self.up2 = UpBlock(num_channel * 3, num_channel * 2)
#         self.up3 = UpBlock(num_channel * 2, num_channel)
#         self.up4 = UpBlock(num_channel, 3, is_last=True)

#     def forward(self, inputs):
#         d1 = self.down1(inputs)
#         d2 = self.down2(d1)
#         d3 = self.down3(d2)
#         d4 = self.down4(d3)
#         res = self.res_blocks(d4)
#         u1 = self.up1(res)
#         u2 = self.up2(u1 + d3)
#         u3 = self.up3(u2 + d2)
#         output = self.up4(u3 + d1)
#         return output

# if __name__ == '__main__':

# #if _name_ == '_main_':
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=12000)


import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import torch
import torch.nn as nn
from werkzeug.utils import secure_filename
import rembg
from typing import List
import urllib.parse


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


# @app.post("/toonify")
# async def toonify(files: List[UploadFile] = File(...)):
#     responses = []
#     for file in files:
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             with open(file_path, "wb") as buffer:
#                 buffer.write(await file.read())
#             process_file(file_path, filename)
#             for i in range(5):
#                 cartoon_path = os.path.join(app.config['DOWNLOAD_FOLDER'], 'cartoon', f"cartoon_{i+1}_" + filename)
#                 view_link = f"/view?path={cartoon_path}"
#                 download_link = f"/download?path={cartoon_path}"
#                 responses.append(
#                     {
#                         "filename": f"cartoon_{i+1}_" + filename,
#                         "view_link": view_link,
#                         "download_link": download_link
#                     }
#                 )
#         else:
#             return {"error": "Invalid file format. Please upload an image file (PNG, JPG, JPEG)."}
#     return responses

@app.post("/toonify")
async def toonify(files: List[UploadFile] = File(...)):
    responses = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            process_file(file_path, filename)
            for i in range(5):
                cartoon_path = os.path.join(app.config['DOWNLOAD_FOLDER'], 'cartoon', f"cartoon_{i+1}_" + filename)
                view_link = f"/view?path={cartoon_path}"
                download_link = f"/download?path={cartoon_path}"
                share_link = f"http://localhost:8000/share?file_name=modified_{i+1}_" + filename
                responses.append(
                    {
                        "filename": f"cartoon_{i+1}_" + filename,
                        "view_link": view_link,
                        "download_link": download_link,
                        #"share_link": share_link  # Added share link
                    }
                )
        else:
            return {"error": "Invalid file format. Please upload an image file (PNG, JPG, JPEG)."}
    return responses

@app.get("/download")
async def download(path: str):
    return FileResponse(path, filename=os.path.basename(path))


# @app.get("/view")
# async def view_image(path: str):
#     image_path = os.path.abspath(path)
#     html_content = f"""
#     <html>
#     <body>
#         <img src="file://{image_path}" alt="image" width="500">
#         <button onclick="downloadImage()">Download</button>
#         <script>
#         function downloadImage() {{
#             window.location.href = "{path}";
#         }}
#         </script>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content, status_code=200)

import base64

# @app.get("/view")
# async def view_image(path: str):
#     image_path = os.path.abspath(path)

#     # Read the image and convert it to base64
#     with open(image_path, "rb") as file:
#         image_data = file.read()
#         encoded_image = base64.b64encode(image_data).decode("utf-8")

#     html_content = f"""
#     <html>
#     <body>
#         <img src="data:image/jpeg;base64,{encoded_image}" alt="image" width="500">
#         <button onclick="downloadImage()">Download</button>
#         <script>
#         function downloadImage() {{
#             window.location.href = "{path}";
#         }}
#         </script>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content, status_code=200)



import shutil
# @app.get("/share")
# async def share(file_name: str):
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
#     return FileResponse(file_path, filename=file_name)
import urllib.parse

@app.get("/share")
async def share(file_name: str):
    modified_file_name = urllib.parse.unquote(file_name)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], modified_file_name)
    return FileResponse(file_path, filename=modified_file_name)

# @app.get("/view")
# async def view_image(path: str):
#     image_path = os.path.abspath(path)
#     download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], "index.html")

#     # Read the image and convert it to base64
#     with open(image_path, "rb") as file:
#         image_data = file.read()
#         encoded_image = base64.b64encode(image_data).decode("utf-8")

#     html_content = f"""
#     <html>
#     <body>
#         <img src="data:image/jpeg;base64,{encoded_image}" alt="image" width="500">
#         <button onclick="downloadImage()">Download</button>
#         <script>
#         function downloadImage() {{
#             window.location.href = "{path}";
#         }}
#         </script>
#     </body>
#     </html>
#     """

#     # Save the HTML content to a file
#     with open(download_path, "w") as file:
#         file.write(html_content)

#     # Move the file to the appropriate directory
#     destination_path = os.path.join(DIR_PATH, download_path)
#     shutil.move(download_path, destination_path)

#     return FileResponse(destination_path, filename="index.html")
# @app.get("/view")
# async def view_image(path: str):
#     image_path = os.path.abspath(path)
#     download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], "index.html")

#     # Read the image and convert it to base64
#     with open(image_path, "rb") as file:
#         image_data = file.read()
#         encoded_image = base64.b64encode(image_data).decode("utf-8")

#     html_content = f"""
#     <html>
#     <body>
#         <img src="data:image/jpeg;base64,{encoded_image}" alt="image" width="500">
#         <button onclick="downloadImage()">Download</button>
#         <script>
#         function downloadImage() {{
#             window.location.href = "{path}";
#         }}
#         </script>
#     </body>
#     </html>
#     """

#     # Save the HTML content to a file
#     with open(download_path, "w") as file:
#         file.write(html_content)

#     # Move the file to the appropriate directory
#     destination_path = os.path.join(DIR_PATH, download_path)
#     shutil.move(download_path, destination_path)

#     return FileResponse(destination_path, filename="index.html")
# @app.get("/view")
# async def view_image(path: str):
#     image_path = os.path.abspath(path)
#     download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], "index.html")

#     # Read the image and convert it to base64
#     with open(image_path, "rb") as file:
#         image_data = file.read()
#         encoded_image = base64.b64encode(image_data).decode("utf-8")

#     html_content = f"""
#     <html>
#     <body>
#         <img src="data:image/jpeg;base64,{encoded_image}" alt="image" width="500">
#         <button onclick="downloadImage('{path}')">Download</button>
#         <script>
#         function downloadImage(path) {{
#             window.location.href = "/download?path=" + encodeURIComponent(path);
#         }}
#         </script>
#     </body>
#     </html>
#     """

#     # Save the HTML content to a file
#     with open(download_path, "w") as file:
#         file.write(html_content)

#     # Move the file to the appropriate directory
#     destination_path = os.path.join(DIR_PATH, download_path)
#     shutil.move(download_path, destination_path)

#     return FileResponse(destination_path, filename="index.html")
# @app.get("/view")
# async def view_image(path: str):
#     image_path = os.path.abspath(path)
#     download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], "index.html")

#     # Read the image and convert it to base64
#     with open(image_path, "rb") as file:
#         image_data = file.read()
#         encoded_image = base64.b64encode(image_data).decode("utf-8")

#     html_content = f"""
#     <html>
#     <body>
#         <img src="data:image/jpeg;base64,{encoded_image}" alt="image" width="500">
#         <button onclick="downloadImage('{path}')">Download</button>
#         <script>
#         function downloadImage(path) {{
#             window.location.href = "/download?path=" + encodeURIComponent(path);
#         }}
#         </script>
#     </body>
#     </html>
#     """

#     # Save the HTML content to a file
#     with open(download_path, "w") as file:
#         file.write(html_content)

#     # Move the file to the appropriate directory
#     destination_path = os.path.join(DIR_PATH, download_path)
#     shutil.move(download_path, destination_path)

#     return FileResponse(destination_path, filename="index.html")
# import base64
# import shutil
# import urllib.parse
# @app.get("/view")
# async def view_image(path: str):
#     image_path = os.path.abspath(path)
#     download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], "index.html")

#     # Read the image and convert it to base64
#     with open(image_path, "rb") as file:
#         image_data = file.read()
#         encoded_image = base64.b64encode(image_data).decode("utf-8")

#     html_content = f"""
#     <html>
#     <body>
#         <img src="data:image/jpeg;base64,{encoded_image}" alt="image" width="500">
#         <button onclick="downloadImage('{path}')">Download</button>
#         <script>
#         function downloadImage(path) {{
#             window.location.href = "/download?path=" + encodeURIComponent(path);
#         }}
#         </script>
#     </body>
#     </html>
#     """

#     # Save the HTML content to a file
#     with open(download_path, "w") as file:
#         file.write(html_content)

#     # Move the file to the appropriate directory
#     destination_path = os.path.join(DIR_PATH, download_path)
#     shutil.move(download_path, destination_path)

#     return FileResponse(destination_path, filename="index.html")
import base64
import shutil
import urllib.parse
from fastapi.responses import FileResponse

@app.get("/view")
async def view_image(path: str):
    image_path = os.path.abspath(path)

    # Read the image and convert it to base64
    with open(image_path, "rb") as file:
        image_data = file.read()

    # Create a response with the image data and appropriate MIME type
    mime_type = "image/jpeg" if image_path.endswith(".jpg") else "image/png"
    return Response(content=image_data, media_type=mime_type)

# @app.get("/view")
# async def view_image(path: str):
#     image_path = os.path.abspath(path)

#     # Read the image and convert it to base64
#     with open(image_path, "rb") as file:
#         image_data = file.read()
#         encoded_image = base64.b64encode(image_data).decode("utf-8")

#     html_content = f"""
#     <html>
#     <body>
#         <img src="data:image/jpeg;base64,{encoded_image}" alt="image" width="500">
#         <a href="{path}" download>Download</a>
#     </body>
#     </html>
#     """

#     download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], "index.html")

#     # Save the HTML content to a file
#     with open(download_path, "w") as file:
#         file.write(html_content)

#     # Move the file to the appropriate directory
#     destination_path = os.path.join(DIR_PATH, download_path)
#     shutil.move(download_path, destination_path)

#     return FileResponse(destination_path, filename="index.html")

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
from starlette.responses import StreamingResponse

# @app.get("/download")
# async def download(path: str):
#     file_path = os.path.abspath(path)
#     filename = os.path.basename(file_path)

#     return StreamingResponse(open(file_path, "rb"), media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename={filename}"})
import mimetypes

# @app.get("/download")
# async def download(path: str):
#     filename = os.path.basename(path)
#     mime_type, _ = mimetypes.guess_type(filename)
#     headers = {
#         "Content-Disposition": f"attachment; filename={filename}",
#         "Content-Type": mime_type,
#     }
#     return FileResponse(path, headers=headers)
@app.get("/download")
async def download(path: str):
    return FileResponse(path, filename=os.path.basename(path), headers={"Content-Disposition": "attachment"})

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=12000)
