# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
from fastapi import FastAPI, File, Form, Response , Request, UploadFile
from fastapi.responses import HTMLResponse,StreamingResponse,JSONResponse
from typing_extensions import Annotated
from typing import Optional,List
from enum import Enum
import torchaudio
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from fastapi.responses import FileResponse
from io import BytesIO
import requests
from pydub import AudioSegment
import aiofiles
import wave
import numpy as np
import urllib.parse
class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"

model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"))
m.eval()

regex = r"<\|.*\|>"
# 用于存储所有音频片段的全局列表
audio_segments = []
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """

@app.post("/api/v1/asr")
async def turn_audio_to_text(files: Annotated[List[bytes], File(description="wav or mp3 audios in 16KHz")], keys: Annotated[str, Form(description="name of each audio joined with comma")], lang: Annotated[Language, Form(description="language of audio content")] = "auto"):
    audios = []
    audio_fs = 0
    for file in files:
        file_io = BytesIO(file)
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)
        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)
        file_io.close()
    if lang == "":
        lang = "auto"
    if keys == "":
        key = ["wav_file_tmp_name"]
    else:
        key = keys.split(",")
    res = m.inference(
        data_in=audios,
        language=lang, # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=False,
        key=key,
        fs=audio_fs,
        **kwargs,
    )
    if len(res) == 0:
        return {"result": []}
    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)#纯文本
        it["text"] = rich_transcription_postprocess(it["text"])#带表情的文本
    return {"result": res[0]}

@app.post("/api/v1/asr1")
async def turn_audio_to_text(files: Annotated[List[bytes], File(description="wav or mp3 audios in 16KHz")]):
    audios = []
    audio_fs = 0
    for file in files:
        file_io = BytesIO(file)
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)
        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)
        file_io.close()
    # if lang == "":
    #     lang = "auto"
    # if keys == "":
    #     key = ["wav_file_tmp_name"]
    # else:
    #     key = keys.split(",")
    res = m.inference(
        data_in=audios,
        language="zh", # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=False,
        key="wav_file_tmp_name",
        fs=audio_fs,
        **kwargs,
    )
    if len(res) == 0:
        return {"result": []}
    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)#纯文本
        it["text"] = rich_transcription_postprocess(it["text"])#带表情的文本
    return {"result": res[0]}

@app.post("/api/v1/asr2")
async def turn_audio_to_text(files: Annotated[List[bytes], File(description="wav or mp3 audios in 16KHz")], keys: Annotated[str, Form(description="name of each audio joined with comma")], lang: Annotated[Language, Form(description="language of audio content")] = "auto"):
    audios = []
    audio_fs = 0
    for file in files:
        file_io = BytesIO(file)
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)
        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)
        file_io.close()
    if lang == "":
        lang = "auto"
    if keys == "":
        key = ["wav_file_tmp_name"]
    else:
        key = keys.split(",")     
    res = m.inference(
        data_in=audios,
        language=lang, # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=False,
        key=key,
        fs=audio_fs,
        **kwargs,
    )
    if len(res) == 0:
        return {"result": []}
    for it in res[0]:
        text = re.sub(regex, "", it["text"], 0, re.MULTILINE)#纯文本
    api_url = "http://192.168.6.50:3001/api/v1/workspace/local_genius/chat"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer 2G6X7NR-R0E4AET-KGR57GR-NTD1QKK",
        "Content-Type": "application/json"
    }
    data = {
    "message": text,
    "mode": "chat"
            }
    response = requests.post(api_url, json=data, headers=headers).json()["textResponse"]
    # 定义请求的URL
    url = "http://192.168.6.50:9880/"
    # 定义请求参数
    params = {
        "text": response,
        "media_type": "wav",
        "seed": 2581,
        "streaming": 1,
        "roleid": None
    }
    # 发起GET请求
    response = requests.get(url, params=params, stream=True)
    
    return StreamingResponse(response.iter_content(chunk_size=1024), media_type="audio/wav")
@app.post("/api/v1/asr3")
async def turn_to_audio(text: str = "你好"):
    api_url = "http://192.168.6.50:3001/api/v1/workspace/local_genius/chat"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer 2G6X7NR-R0E4AET-KGR57GR-NTD1QKK",
        "Content-Type": "application/json"
    }
    data = {
    "message": text,
    "mode": "chat"
    }
    response = requests.post(api_url, json=data, headers=headers).json()["textResponse"]
    # 定义请求的URL
    url = "http://127.0.0.1:9880/"
    # 定义请求参数
    params = {
        "text": response,
        "media_type": "wav",
        "seed": 2581,
        "streaming": 1,
        "roleid": None
    }
    # 发起GET请求
    response = requests.get(url, params=params, stream=True)
    
    return StreamingResponse(response.iter_content(chunk_size=1024), media_type="audio/wav")
@app.get("/api/v1/asr4/")
async def turn_to_audio(text: str = None,top:int=0):
    speaker=["hdj","lxy",'lcy',"金牌讲师","蜡笔小新","叶奈法","jok老师","Keira","阿星","步非烟","元气女声", '剑魔', '剑魔2','发烧的冰箱']
    api_url = "http://127.0.0.1:3001/api/v1/workspace/local_genius/chat"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer 2G6X7NR-R0E4AET-KGR57GR-NTD1QKK",
        "Content-Type": "application/json"
    }
    data = {
    "message": text,
    "mode": "chat"
    }
    response = requests.post(api_url, json=data, headers=headers).json()["textResponse"]
    # 定义请求的URL
    url = "http://127.0.0.1:9880/"
    # 定义请求参数
    params = {
        "text": response,
        "speaker":speaker[top] ,
        "streaming": 1,
    }
    # 发起GET请求
    response = requests.get(url, params=params, stream=True)
    return StreamingResponse(response.iter_content(chunk_size=1024), media_type="audio/mp3")
    
@app.post("/api/v1/asr5")
async def turn_audio_to_text(files: Annotated[List[UploadFile], File()]):
    for file in files:
        contents = await file.read()
        with open(r"C:\Users\hijie\Desktop\out.wav", 'wb') as f:
            f.write(contents)
    return {"message": "Files saved successfully"}
@app.post("/api/v1/asr6")
async def turn_audio_to_text(request: Request):
    contents = await request.body()
    audio = AudioSegment(
        data=contents,
        sample_width=2,  # 16-bit
        frame_rate=22000,  # 16kHz
        channels=1  # mono
    )
    # audio = audio.set_frame_rate(16000)
    audio_segments.append(audio)
    
    return {"message": "Audio chunk received successfully"}

@app.get("/api/v1/merge")
async def merge_audio():
    # 合并所有音频片段
    combined = AudioSegment.silent(duration=0)
    for segment in audio_segments:
        combined += segment
    # 保存合并后的WAV文件
    output_file = r"C:\Users\hi jie\Desktop\out.wav"
    combined.export(output_file, format="wav")

    # 加载合并后的音频文件
    with open(output_file, 'rb') as f:
        audio_data = f.read()

    # 使用 BytesIO 加载音频数据
    file_io = BytesIO(audio_data)
    data_or_path_or_list, audio_fs = torchaudio.load(file_io)
    data_or_path_or_list = data_or_path_or_list.mean(0)

    # 进行音频到文本的转换
    res = m.inference(
        data_in=[data_or_path_or_list],
        language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=False,
        key="wav_file_tmp_name",
        fs=audio_fs,
        **kwargs,
    )

    if len(res) == 0:
        return {"result": []}

        # 对转换结果进行处理
    result = {}
    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)  # 纯文本
        it["text"] = rich_transcription_postprocess(it["text"])  # 带表情的文本
        it["text_url"] = urllib.parse.quote(it["text"])  # URL 编码的纯文本
        result={"text": it["text"], "text_url": it["text_url"]}

    # 清空音频片段
    audio_segments.clear()

    # 返回 JSON 格式的响应
    return result
@app.get("/audio")
async def get_audio():
    return FileResponse(r"C:\Users\hi jie\Desktop\1234.wav")


