from fastapi import FastAPI, WebSocket
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, pipeline
import torch
import io
import soundfile as sf
import numpy as np
import tempfile
import os

app = FastAPI()

# Load local LLM
tokenizer_llm = AutoTokenizer.from_pretrained("microsoft/phi-2")
model_llm = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

# Load local TTS model
processor_tts = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model_tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load local Whisper model
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base") #or a larger model.

device = "cuda" if torch.cuda.is_available() else "cpu"
print('device:', device)
model_llm.to(device)
model_tts.to(device)
vocoder.to(device)

async def transcribe_audio(audio_data):
    try:
        audio_bytes = bytes(audio_data)
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        temp_file_path = os.path.join(desktop_path, "audio_temp.webm") #predictable name.
        print('temp_file_path=>>>>>>>>>', temp_file_path)
        with open(temp_file_path, "wb") as temp_audio_file:
            print('Inside file create and read', temp_file_path)
            temp_audio_file.write(audio_bytes)

        print(f"File exists before whisper: {os.path.exists(temp_file_path)}")

        transcript = transcriber(temp_file_path)["text"]

        print(f"File exists after whisper: {os.path.exists(temp_file_path)}")

        os.remove(temp_file_path)
        return transcript
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

async def get_llm_response(text):
    try:
        inputs = tokenizer_llm(text, return_tensors="pt").to(device)
        outputs = model_llm.generate(**inputs, max_new_tokens=200)
        response = tokenizer_llm.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"LLM error: {e}")
        return None

async def text_to_speech(text):
    try:
        inputs = processor_tts(text=text, return_tensors="pt")
        speech = model_tts.generate_speech(inputs["input_ids"].to(device), vocoder=vocoder).cpu().numpy()
        return speech.tobytes()
    except Exception as e:
        print(f"TTS error: {e}")
        return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            audio_data = data["audio"]
            transcription = await transcribe_audio(audio_data)
            if transcription:
                llm_response = await get_llm_response(transcription)
                if llm_response:
                    speech_audio = await text_to_speech(llm_response)
                    await websocket.send_bytes(speech_audio)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)