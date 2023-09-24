from flask import Flask, Blueprint, render_template, jsonify, request, send_file
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import numpy as np
import glob 
import cv2
import json
import base64
import re
import os
import openai
import torch
import deepl
import time

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

bp = Blueprint('create_ill', __name__, url_prefix='/')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# #1. huggingface
# hugging_token = 'token' #해니

# #2. deepl
# translator = deepl.Translator("key") #해니

# #3. openai
# openai.api_key = "key"  #해니
# model = "gpt-3.5-turbo"

# #Stable Diffusion

# # 서버 컴퓨터
# # pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
# #                                              torch_dtype=torch.float32, 
# #                                              use_safetensors=True, 
# #                                              variant="fp16").to(device)

# # 일반 노트북
# pipeline = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',
#                                                revision='fp16',
#                                                torch_dtype=torch.float32,
#                                                use_auth_token=hugging_token
#                                                ).to(device)

# #4. insight face
# app = FaceAnalysis(name='buffalo_l')
# app.prepare(ctx_id=0, det_size=(640, 640))

# swapper = insightface.model_zoo.get_model('inswapper_128.onnx',
#                                           download=False,
#                                           download_zip=False)


# @bp.route('/step1', methods=['POST', 'GET'])
# def basic_charactor():
#     if request.method == 'POST':

#         data = request.get_json()  # JSON 데이터 추출

#         # 추출된 JSON 데이터 사용
#         title = data.get('title')
#         name = data.get('name')
#         gender = data.get('gender')
#         appearance = data.get('appearance')
#         personality = data.get('personality')
#         genre = data.get('genre')

#         # Translate the text using the Deepl API
#         result = translator.translate_text(f"{name} : 성별은 {gender},생김새는 {appearance}, 성격은 {personality}, 장르는 {genre}, 웹툰그림체, 한 명, 정면 응시, 상반신, 배경 없음", target_lang="en-us")

#         # Set up the prompt
#         prompt = result.text

#         #----Face detection 하는 코드 추가, detect 안되면 다시 돌리기. 일단 한번은 실행되도록. 

#         # Generate the image
#         image = pipeline(prompt).images[0]

#         # Generate a unique image filename
#         timestamp = int(time.time())
#         image_filename = f'static/novel/basic/{title}_{name}_{timestamp}.png'  # Save to the static/image folder

#         # Save the image to the specified path
#         image.save(image_filename, format='PNG')

#         # Convert the image to Base64
#         with open(image_filename, "rb") as image_file:
#             encoded_image = base64.b64encode(image_file.read()).decode()

#         # Return a JSON response with the Base64-encoded image and image filename
#         response_data = {
#             "image_base64": encoded_image,
#             "image_filename": image_filename
#         }

#         return jsonify(response_data)
    


# @bp.route('/step4', methods=['POST', 'GET'])  #삽화 생성 후 저장 버튼 눌렀을 때 처리, 넘긴 이미지는 마지막 단계에서 보여주기 위해 저장.
# def face_detection():
#     if request.method == 'POST':
#         try:
#             # 클라이언트로부터 JSON 데이터를 받아옴
#             data = request.json

#             # Base64로 인코딩된 이미지 데이터를 디코딩
#             base64_image = data.get('image', '') #생성된 소설 삽화

#             if not base64_image:
#                 return jsonify({'error': '이미지 데이터가 없습니다.'}), 400

#             # Base64 디코딩
#             image_bytes = base64.b64decode(base64_image)

#             # Bytes를 NumPy 배열로 변환
#             image_np = np.frombuffer(image_bytes, np.uint8)

#             # NumPy 배열을 OpenCV 이미지로 변환
#             image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

#             if image is None:
#                 return jsonify({'error': '이미지를 읽을 수 없습니다.'}), 400

#             # 얼굴 검출
#             faces = app.get(image)

#             detect_num = len(faces)
#             detect_images = []  # 인식된 얼굴 이미지들을 저장할 리스트

#             for i, face in enumerate(faces):
#                 bbox = face['bbox']
#                 bbox = [int(b) for b in bbox]
#                 timestamp = int(time.time())
#                 face_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1]
#                 face_encoded = base64.b64encode(cv2.imencode('.png', face_image)[1]).decode()  # 이미지를 Base64로 인코딩
#                 detect_images.append(face_encoded)

#             response_data = {
#                 'detect_num': detect_num,
#                 'detect_images': detect_images
#             }

#             return jsonify(response_data), 200

#         except Exception as e:
#             return jsonify({'error': str(e)}), 400
    

# @bp.route('/step5', methods=['POST', 'GET']) #이미지값들 받아서 순서대로 얼굴 교체. 
# def face_swap():
#     if request.method == 'POST':
#         try:
#             # JSON 데이터에서 이미지 리스트와 일러스트 이미지를 받아옴
#             data = request.get_json()

#             # 'b_list'에서 여러장의 인코딩된 이미지 리스트를 가져옴
#             encoded_images = data.get('b_list', [])

#             # 'illustration'에서 한장의 인코딩된 이미지를 가져옴
#             encoded_illustration = data.get('illustration', '')

#             # 일러스트 이미지 디코드
#             illustration_bytes = base64.b64decode(encoded_illustration)
#             illustration_np = np.frombuffer(illustration_bytes, np.uint8)
#             illustration = cv2.imdecode(illustration_np, cv2.IMREAD_COLOR)

#             # 얼굴 디텍션
#             faces = app.get(illustration)

#             # 인코딩된 이미지를 디코드하고 얼굴 스왑
#             for i, face in enumerate(faces):
#                 if i < len(encoded_images):
#                     encoded_image = encoded_images[i]
#                     image_bytes = base64.b64decode(encoded_image)
#                     image_np = np.frombuffer(image_bytes, np.uint8)
#                     face1 = app.get(cv2.imdecode(image_np, cv2.IMREAD_COLOR))[0]
#                     face2 = faces[i]
#                     illustration = swapper.get(illustration, face1, face2, paste_back=True)

#             # 결과 이미지를 base64로 인코딩
#             _, illustration_encoded = cv2.imencode('.jpg', illustration)
#             illustration_base64 = base64.b64encode(illustration_encoded).decode('utf-8')

#             return jsonify({'result_image': illustration_base64})

#         except Exception as e:
#             return jsonify({'error': str(e)}), 400