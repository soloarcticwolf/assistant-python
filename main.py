from fastapi import FastAPI, WebSocket
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, pipeline
import torch
import io
import soundfile as sf
import numpy as np
import tempfile
import os
from datasets import load_dataset
import scipy.io.wavfile

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

# Load speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0) #you can change the index to change speaker.
speaker_embeddings = speaker_embeddings.to(device)


print('CUDA AVAILABLE? :', torch.cuda.is_available())
print('DEVICE NAME', torch.cuda.get_device_name(0))
print('*****Device', device)

async def transcribe_audio(audio_data):
    try:
        audio_bytes = bytes(audio_data)

        # Create a temporary directory in the user's home directory
        temp_dir = os.path.join(os.path.expanduser("~"), "voice_assistant_temp")
        os.makedirs(temp_dir, exist_ok=True)  # Create if it doesn't exist

        # Create a temporary file within that directory
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False, dir=temp_dir) as temp_audio_file:
            temp_file_path = temp_audio_file.name
            temp_audio_file.write(audio_bytes)

        print(f"Temporary file created: {temp_file_path}")
        print(f"File exists before whisper: {os.path.exists(temp_file_path)}")

        transcript = transcriber(temp_file_path)["text"]

        print('*******transcript', transcript)

        print(f"File exists after whisper: {os.path.exists(temp_file_path)}")

        os.remove(temp_file_path)  # Delete the file
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
    print('RESPONSE TEXT=>',text)
    try:
        inputs = processor_tts(text=text, return_tensors="pt")
        if inputs["input_ids"].shape[1] > 600:
            print("Warning: Input text truncated to 600 tokens.")
            inputs["input_ids"] = inputs["input_ids"][:, :600]
            inputs["attention_mask"] = inputs["attention_mask"][:, :600]
        speech = model_tts.generate_speech(inputs["input_ids"].to(device), speaker_embeddings, vocoder=vocoder)
        speech_cpu = speech.cpu().numpy() # Move to CPU before converting to NumPy
        
        # Encode to WAV
        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, rate=16000, data=speech_cpu) #16000 is the sample rate.
        wav_bytes = wav_buffer.getvalue()
        return wav_bytes
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
                    if speech_audio: #check that speech_audio is not None.
                        await websocket.send_bytes(speech_audio)
                    else: print('>>>>>>>>>>>>No speech audio!')
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)