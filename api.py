import os
import json
import wave
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()

app = FastAPI(title="Shelf Voice API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN   = os.getenv("HF_TOKEN")
REPO_ID    = "AmaZed007/Vosk_Voice_Recognition_model"
LOCAL_DIR  = "./models"
vosk_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vosk_model
    model_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=LOCAL_DIR,
        token=HF_TOKEN,
    )

    vosk_dir = _find_vosk_dir(model_path)
    if not vosk_dir:
        raise RuntimeError(
            f"Could not find Vosk model folder inside {model_path}. "
            "Make sure your HF repo contains the Vosk model files."
        )
    vosk_model = Model(vosk_dir)

    yield

app = FastAPI(title="Shelf Voice API", lifespan=lifespan)

def _find_vosk_dir(base: str):
    for root, dirs, files in os.walk(base):
        if "am" in dirs and "conf" in dirs:
            return root
    if os.path.isdir(os.path.join(base, "am")):
        return base
    return None



def convert_to_pcm16k(audio_bytes: bytes) -> bytes:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with wave.open(tmp_path, "rb") as wf:
            n_ch = wf.getnchannels()
            rate = wf.getframerate()
            raw  = wf.readframes(wf.getnframes())

        os.remove(tmp_path)

        audio_np = np.frombuffer(raw, dtype=np.int16)

        if n_ch == 2:
            audio_np = audio_np.reshape(-1, 2).mean(axis=1).astype(np.int16)

        if rate != 16000:
            new_len  = int(len(audio_np) * 16000 / rate)
            audio_np = np.interp(
                np.linspace(0, len(audio_np), new_len),
                np.arange(len(audio_np)),
                audio_np
            ).astype(np.int16)

        return audio_np.tobytes()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio conversion error: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": vosk_model is not None}


@app.post("/voice")
async def voice_command(audio: UploadFile = File(...)):
    if vosk_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    raw_bytes  = await audio.read()
    pcm        = convert_to_pcm16k(raw_bytes)

    rec = KaldiRecognizer(vosk_model, 16000)
    rec.AcceptWaveform(pcm)
    result     = json.loads(rec.FinalResult())
    transcript = result.get("text", "").lower().strip()

    print(f"Transcript: '{transcript}'")
    return {"transcript": transcript}