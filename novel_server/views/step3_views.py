from flask import Flask, Blueprint, render_template, send_file, request, jsonify
import deepl
from flask import Blueprint, Flask, jsonify, request
import openai
import requests
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from PIL import Image
import os

bp = Blueprint('step3', __name__, url_prefix='/')
 
# # api
# translator = deepl.Translator('')  ### 삭제

# model = "gpt-3.5-turbo"

# openai.api_key = ""  ### 삭제

# hugging_token = '' ### 삭제

# # cuda 설정
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # draw_by = 'Galan Pang'

# # 번역  
# def trans_ko_eng(input):
#     result = translator.translate_text(input, target_lang='EN-US')
#     return result

# # 영어로 번역해서 이미지 출력
# @bp.route('/step3', methods=['POST'])
# def eng_show():
    
#     data = request.json

#     genre = data.get('genre', '')
#     character = data.get('character', '')
#     episode_plot = data.get('episode_plot', '')
#     whole_plot = data.get('whole_plot', '')
#     input = data.get('input', '')

#     trans_genre = str(trans_ko_eng(genre))
#     trans_character = str(trans_ko_eng(character))
#     trans_episode_plot = str(trans_ko_eng(episode_plot))
#     trans_whole_plot = str(trans_ko_eng(whole_plot))
#     trans_input = str(trans_ko_eng(input))

#     # stable diffusion 프롬프트 작성
#     def summarization(trans_genre, trans_character, trans_episode_plot, trans_whole_plot, trans_input):
    
#         messages = [          
#                     {"role": "system", "content": "You should not let the prompt exceed 70 tokens"},      
#                     {"role": "system", "content": f"Please extract extra important elements separated by commas, such as verbs, subjects from {trans_episode_plot}, {trans_character}'s perspective and you must include {trans_input}"},
#                     {"role": "system", "content": f"Please extract the extra important elements separated by commas, such as verbs, subjects and must include {trans_character}"},                  
#             ]
    
#         response = openai.ChatCompletion.create(
#         model = model,
#         messages=messages,
#         max_tokens = 3000
    
#         )

#         result = response['choices'][0]['message']['content']

#         return result
    
#     result = summarization(trans_genre, trans_character, trans_episode_plot, trans_whole_plot, trans_input)

#     # 서버 아닐 시
#     # pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',
#                                                 #    revision='fp16',
#                                                 #    torch_dtype=torch.float32,
#                                                 #    use_auth_token=hugging_token
#                                                 #    ).to(device)
    
    
#     # 이미지 파일 저장 디렉토리  => 변경 필요
#     save_directory = 'D:/toon_novel/static/image/novel/romancefantasy'

#     # 이미지 파일 저장 번호 초기화
#     image_number = 1

#     pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, variant="fp16").to(device)

#     # 이미지 파일 이름 생성 및 확인  => 경로 수정하기
#     while True:
#         file_name = f'Todd Schorr{image_number}.png'
#         file_path = os.path.join(save_directory, file_name)

#         # 이미 파일이 존재하지 않는 경우에만 저장
#         if not os.path.exists(file_path):

#             # 번역 수행
#             prompt = result + f',art by Todd Schorr, approaching perfection, masterpiece, best quality, intricate,  sharp, focused, not blurry'
#             print(f'\n\nprompt : {prompt}')

#             image = pipe(prompt).images[0]

#             # 이미지를 파일로 저장
#             image.save(file_path.replace('\\', '/')) 
#             print(f"이미지를 {file_name} 파일로 저장했습니다.")

#             break    

#         image_number += 1
    
#     # image_url = '/static/image/novel/complete1.png'

#     # response_data = {
#     # 'message': '이미지 생성 및 저장 성공',
#     # 'image_url': image_url
#     # } 
#     # return jsonify(response_data)
    
#      # 서버에 파일 'rb'(바이너리 리드)방식으로 엶
#     with open(file_path, 'rb') as file:
#         files = {'file' : (file_name, file)}
#         server_url = '서버 주소'
#         response = requests.post(server_url, files=files)
#         if response.status_code == 200:
#             print('파일 전송 성공')
#         else:
#             print('파일 전송 실패')
    
#     return jsonify({'image_url': file_path})
    
    
# @bp.route('/show_img', methods=['POST'])
# def img_show():
#     data = request.files['file']
    
#     # 파일 없는 경우
#     if data is None:
#         return '파일이 존재하지 않습니다.'
        
#     else:  # 파일 있는 경우
#         return '이미지 받기 성공'

# # @bp.route('/trans_eng', methods=['POST'])
# # def eng_show():
# #     data = request.json
# #     input_text = data.get('english', '')  
# #     trans_output = trans_ko_eng(input_text)
# #     return jsonify({"번역한 내용": trans_output.text})
