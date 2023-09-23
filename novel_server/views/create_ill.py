from flask import Flask, Blueprint, render_template, jsonify, request, send_file
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import numpy as np
import requests
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
import tempfile
import shutil

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

bp = Blueprint('create_ill', __name__, url_prefix='/')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#1. huggingface
hugging_token = '' #해니

#2. deepl
translator = deepl.Translator("") #해니

#3. openai
openai.api_key = ""  #해니
model = "gpt-3.5-turbo"

#Stable Diffusion

# 서버 컴퓨터
# pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
#                                              torch_dtype=torch.float32, 
#                                              use_safetensors=True, 
#                                              variant="fp16").to(device)

# 일반 노트북
pipeline = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',
                                               revision='fp16',
                                               torch_dtype=torch.float32,
                                               use_auth_token=hugging_token
                                               ).to(device)

#4. insight face
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model('inswapper_128.onnx',
                                          download=False,
                                          download_zip=False)


@bp.route('/step1', methods=['POST', 'GET'])
def basic_charactor():
    if request.method == 'POST':

        data = request.get_json()  # JSON 데이터 추출

        # 추출된 JSON 데이터 사용
        title = data.get('title')
        name = data.get('name')
        gender = data.get('gender')
        appearance = data.get('appearance')
        personality = data.get('personality')
        genre = data.get('genre')

        # Translate the text using the Deepl API
        result = translator.translate_text(f"{name} : 성별은 {gender},생김새는 {appearance}, 성격은 {personality}, 장르는 {genre}, 웹툰그림체, 정면 응시, 상반신", target_lang="en-us")

        # Set up the prompt
        prompt = result.text

        #----Face detection 하는 코드 추가, detect 안되면 다시 돌리기. 일단 한번은 실행되도록. 
        # len(face)가 0이 아니면 break

        # Generate the image
        image = pipeline(prompt+', single person, no background').images[0]

        # Generate a unique image filename
        timestamp = int(time.time())

        save_directory = 'C:/Users/user/miniforge3/dev/ai-project-novel/novel_server/static/novel/basic'
        file_name = f'{title}_{name}_{timestamp}.png'  # Save to the static/image folder
        file_path = os.path.join(save_directory, file_name)

        # Save the image to the specified path
        image.save(file_path.replace('\\', '/'), format='PNG')
        print(f"이미지를 {file_name} 파일로 저장했습니다.")

        # 서버에 파일 'rb'(바이너리 리드)방식으로 엶
        with open(file_path, 'rb') as file:
            files = {'file' : (file_name, file)}
            server_url = '전송할 서버 주소'
            response = requests.post(server_url, files=files)
            if response.status_code == 200:
                print('파일 전송 성공')
            else:
                print('파일 전송 실패')
    

        return '이미지 전송 성공'
    


@bp.route('/step4', methods=['POST', 'GET'])  #삽화 생성 후 저장 버튼 눌렀을 때 처리, 넘긴 이미지는 마지막 단계에서 보여주기 위해 저장.
def face_detection():
    if request.method == 'POST':
        try:
            # form-data에서 이미지 데이터 추출
            image_file = request.files['image']

            if not image_file:
                return jsonify({'error': '이미지 데이터가 없습니다.'}), 400

            # 이미지 데이터를 OpenCV 형식으로 읽어옴
            image_np = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)


            if image is None:
                return jsonify({'error': '이미지를 읽을 수 없습니다.'}), 400

            # 얼굴 검출
            faces = app.get(image)

            detect_num = len(faces)
            detect_images = []  # 인식된 얼굴 이미지들을 저장할 리스트

            for i, face in enumerate(faces):
                bbox = face['bbox']
                bbox = [int(b) for b in bbox]
                timestamp = int(time.time())
                face_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1]
                face_encoded = base64.b64encode(cv2.imencode('.png', face_image)[1]).decode()  # 이미지를 Base64로 인코딩
                detect_images.append(face_encoded)

            response_data = {
                'detect_num': detect_num,
                'detect_images': detect_images
            }

            return jsonify(response_data), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    

@bp.route('/step5', methods=['POST', 'GET']) #이미지값들 받아서 순서대로 얼굴 교체. 
def face_swap():
    if request.method == 'POST':
        
        # 이미지 파일을 Form Data에서 받아옴
        b_list = request.files.getlist('b_list')
        illustration = request.files['illustration']

        # 일러스트 이미지를 읽어옴
        illustration = cv2.imdecode(np.frombuffer(illustration.read(), np.uint8), cv2.IMREAD_COLOR)

        # 얼굴 디텍션
        faces = app.get(illustration)

        # 인코딩된 이미지를 디코드하고 얼굴 스왑
        for i in range(len(faces)):
            b_list_file = b_list[i]
            face1 = app.get(cv2.imdecode(np.frombuffer(b_list_file.read(), np.uint8), cv2.IMREAD_COLOR))[0]
            face2 = faces[i]

            if i == 0:
                ill_ = illustration.copy()

            ill_ = swapper.get(ill_, face1, face2, paste_back=True)

        save_directory = 'C:/Users/user/miniforge3/dev/ai-project-novel/novel_server/static/novel/result'
        timestamp = int(time.time())
        file_name = f'{timestamp}.png'  # Save to the static/image folder
        file_path = os.path.join(save_directory, file_name)

        # Save the image to the specified path
        #ill_.save(file_path.replace('\\', '/'), format='PNG')
        cv2.imwrite(file_path.replace('\\', '/'), ill_)
        print(f"이미지를 {file_name} 파일로 저장했습니다.")

        # 서버에 파일 'rb'(바이너리 리드)방식으로 엶
        with open(file_path, 'rb') as file:
            files = {'file' : (file_name, file)}
            server_url = '통신할 서버 주소'
            response = requests.post(server_url, files=files)
            if response.status_code == 200:
                print('파일 전송 성공')
            else:
                print('파일 전송 실패')


        return '이미지 전송 성공'