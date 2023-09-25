import json
import re
from flask import Flask, jsonify, render_template, request, Blueprint
import os
import openai

app = Flask(__name__)

bp = Blueprint('step2', __name__, url_prefix='/')

# # 요약
# model = "gpt-3.5-turbo"

# openai.api_key = ""  ### 삭제

# def summarize(input_text):
#             messages = [
#                     {"role": "system", "content": "소설 내용을 한 문장으로 요약해주세요."},
#                     {"role": "user", "content": input_text}
#             ]

#             response = openai.ChatCompletion.create(
#                 model=model,
#                 messages=messages,
#                 max_tokens = 500
#             )

#             return response['choices'][0]['message']['content'] 


# def summarize_text(input_text):
#     if len(input_text) > 3550:  ## 여기 토큰이 처리를 못 하는데 손 보기
#         return '파일의 용량이 너무 커서 처리할 수 없습니다.'
    
#     elif len(input_text) >= 3550: # 내용 길이가 3550 넘는 경우
#         middle = len(input_text) // 2

#         input_text1 = input_text[:middle]
#         input_text2 = input_text[middle:]

#         answer1 = summarize(input_text1)
#         answer2 = summarize(input_text2)

#         return answer1, answer2

#     else: # 내용 길이가 3500 안 넘는 경우
#         messages = [
#                 {"role": "system", "content": "소설 내용을 한 문장으로 요약해주세요."},
#                 {"role": "user", "content": input_text}
#         ]

#         response = openai.ChatCompletion.create(
#             model=model,
#             messages=messages,
#             max_tokens = 500
#         )

#         result = response['choices'][0]['message']['content']

#         return result
        

# # 요약된 모든 문단에 전체 요약 수행
# def total_summarization(input):
#     messages = [
#                 {"role": "system", "content": "소설을 주인공 시선으로 정리해주세요."},
#                 {"role": "system", "content": "한국어로 one sentence 요약해주세요."},
#                 {"role": "user", "content": input}
#         ]

#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         max_tokens = 800
#     )

#     result = response['choices'][0]['message']['content']

#     return result


# # 파일 읽기 (.hwp, .txt, .docx 가능)
# def read_file_with_encoding(file_path):
#     _, file_extension = os.path.splitext(file_path)
#     encoding = 'ISO-8859-1'  # 기본 인코딩
     
#     if file_extension.lower() == '.txt':
#         encoding = 'utf-8'
    
#     with open(file_path, 'r', encoding=encoding) as file:
#         file_contents = file.read()
    
#     return file_contents


# @bp.route('/step2', methods=['POST'])
# def success():
#     if request.method == 'POST':

#         uploaded_file = request.files['file']

#         # 파일 없는 경우
#         if uploaded_file is None:
            # abort(400, '파일이 존재하지 않습니다.')
        
#         else:  # 파일 있는 경우
#             upload_folder = "uploads"  
#             if not os.path.exists(upload_folder):

#                 os.makedirs(upload_folder)
            
#             file_path = os.path.join(upload_folder, uploaded_file.filename)
#             uploaded_file.save(file_path)


#             file_contents = read_file_with_encoding(file_path)
#             pattern = r'(\* \* \*|\*{3})'
#             file_contents = file_contents.replace('\n', ' ')
#             sections = re.split(pattern, file_contents)
#             sections = [section.strip() for section in sections if section.strip()]

#              # re로 나뉜 애 중에서 홀수, 짝수 같은 문단에 존재하게 만듦
#             merged_sections = []

#             for i in range(len(sections)):
#                 if i % 2 == 0 and i < len(sections)-1:
#                     merged_section = sections[i] + ' ' + sections[i+1]
#                     merged_sections.append(merged_section)
#             merged_sections = ",".join(merged_sections)
#             paragraphs = merged_sections.split('\n')

#             # 각 문단 요약
#             summaries = []
            
#             for paragraph in paragraphs:
#                 summary = summarize_text(paragraph)
#                 summaries.append(summary)
                                
#             total_summary = total_summarization('\n'.join(summaries))
#             total_summary = total_summary.replace('\n', ' ')

#             # 파일 내용을 JSON 형식으로 반환
#             response_data = {
#                 'filename': uploaded_file.filename,
#                 'contents': total_summary
#             }

#             return jsonify(response_data)

#             # json 파일 저장
#             # output_file = 'output.json'
#             # with open(output_file, 'w', encoding='utf-8') as output:
#             #     json.dump(response_data, output, ensure_ascii=False, indent=3)

#             # # 한국어로 출력
#             # with open(output_file, 'r', encoding='utf-8') as output:
#             #     output_contents = output.read()
            
#         # 성공 메시지 반환 (json으로)
#         # return json.dumps(response_data, ensure_ascii=False)
        
#         # form 형식으로
#         # return f'result: {response_data}'

#     return "파일 업로드 및 처리에 실패했습니다."
