import webrtcvad
import numpy as np
import whisper
import librosa
import torch
import asyncio
import edge_tts
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


### STEP 1: Voice-to-Text Conversion (with VAD)
def apply_vad(audio, sr, vad_level=2):
    """
    Applied Voice Activity Detection (VAD) to isolate speech from the input audio.

    Parameters:
    - audio: numpy array of audio samples
    - sr: sample rate of the audio
    - vad_level: aggressiveness of the VAD (0-3)

    Returns:
    - vad_audio: bytes object containing audio frames classified as speech
    """
    # Validated VAD level
    if not (0 <= vad_level <= 3):
        raise ValueError("vad_level must be between 0 and 3.")

    vad = webrtcvad.Vad(vad_level)
    audio_bytes = (audio * 32767).astype(np.int16).tobytes()  # Converted to 16-bit PCM
    active_frames = []
    frame_duration = 10  # ms
    frame_size = int(sr * frame_duration / 1000)  # Frame size in samples
    
    for i in range(0, len(audio_bytes), frame_size * 2):
        frame = audio_bytes[i:i + frame_size * 2]
        if len(frame) == frame_size * 2 and vad.is_speech(frame, sr):
            active_frames.append(frame)
    
    if not active_frames:
        raise RuntimeError("No speech detected in the audio.")
    
    return b''.join(active_frames)


def transcribe_with_whisper(vad_audio, sr):
    """
    Transcribed speech from VAD-processed audio using Whisper.

    Parameters:
    - vad_audio: bytes object containing VAD-processed audio
    - sr: sample rate of the audio

    Returns:
    - transcription: Transcribed text from the audio
    """
    model = whisper.load_model("base")

    # Converted bytes back to numpy array for Whisper model processing
    audio_tensor = torch.from_numpy(np.frombuffer(vad_audio, dtype=np.int16).astype(np.float32) / 32768.0)
    result = model.transcribe(audio_tensor.numpy(), fp16=False)
    
    if not result['text'].strip():
        raise RuntimeError("Transcription failed. No text detected.")
    
    return result['text']


### STEP 2: Text Input into LLM
def query_llm(text_input):
    """
    Queried the LLM with transcribed text and generated a response.

    Parameters:
    - text_input: Transcribed text to be input to the LLM

    Returns:
    - restricted_response: The response from the LLM restricted to 2 sentences
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configured the model loading with memory optimization
    bnb_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True, bnb_4bit_compute_dtype=torch.float16)
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

    inputs = tokenizer(text_input, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=50, num_return_sequences=1)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Restricted the output to 2 sentences
    sentences = response.split('. ')
    restricted_response = '. '.join(sentences[:2]) + '.'
    
    return restricted_response


### STEP 3: Text-to-Speech Conversion with Tunable Parameters
async def text_to_speech(text, output_file, pitch=1.0, speed=1.0, voice="en-US-JennyNeural"):
    """
    Converted the input text to speech with adjustable parameters.

    Parameters:
    - text: The text to be converted to speech
    - output_file: Path to the output file where speech will be saved
    - pitch: The pitch adjustment (default is 1.0 for normal pitch)
    - speed: The speed adjustment (default is 1.0 for normal speed)
    - voice: The voice profile to use for TTS
    
    Output:
    - Saved the audio file at the specified path
    """
    try:
        # Converted the speed to a valid string format for the rate, e.g., "+10%" or "-10%"
        rate_percentage = f"{'+' if speed > 1.0 else ''}{int((speed - 1) * 100)}%"

        # Validated the rate
        if not (-100 <= int((speed - 1) * 100) <= 100):
            raise ValueError("Invalid rate. Speed percentage must be between -100% and 100%.")

        # Initialized the TTS engine with the correct parameters
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate_percentage)
        await communicate.save(output_file)
        
        print(f"File saved successfully at: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"An error occurred during TTS conversion: {e}")


# Usage of the Pipeline
async def run_pipeline(audio_file_path, vad_level=2, pitch=1.0, speed=1.0, voice="en-US-JennyNeural"):
    """
    Runs the entire voice query pipeline:
    1. Applied VAD to the input audio
    2. Transcribed the speech to text using Whisper
    3. Inputed the transcription to an LLM to generate a response
    4. Converted the LLM response back to speech with adjustable parameters

    Parameters:
    - audio_file_path: Path to the audio file
    - vad_level: VAD aggressiveness (0-3)
    - pitch: Pitch adjustment for the TTS conversion
    - speed: Speed adjustment for the TTS conversion
    - voice: Voice profile to use for TTS conversion
    """
    try:
        # Step 1: Load and preprocess audio
        print("Loading audio...")
        audio, sr = librosa.load(audio_file_path, sr=16000, mono=True)
        
        # Step 2: Apply Voice Activity Detection (VAD)
        print("Applying VAD...")
        vad_audio = apply_vad(audio, sr, vad_level)
        
        # Step 3: Transcribe using Whisper
        print("Transcribing audio...")
        transcription = transcribe_with_whisper(vad_audio, sr)
        print("Transcribed Text:", transcription)
        
        # Save transcription to a file for verification
        with open("transcription.txt", "w") as f:
            f.write(transcription)
        
        # Step 4: Query LLM and restrict response to 2 sentences
        print("Querying LLM...")
        llm_response = query_llm(transcription)
        print("LLM Response:", llm_response)
        
        # Saved the LLM response to a file for verification
        with open("llm_response.txt", "w") as f:
            f.write(llm_response)
        
        # Step 5: Converted the LLM response to speech
        print("Converting text to speech...")
        output_file = "output.mp3"
        # playsound(output_file)
        await text_to_speech(llm_response, output_file, pitch=pitch, speed=speed, voice=voice)
        
        print("Pipeline completed successfully!")
    except Exception as e:
        print(f"An error occurred in the pipeline: {e}")


# Example call to the pipeline
audio_file_path = r"C:\Users\sabha\Downloads\Music\audio_input_1.mp3"
await run_pipeline(audio_file_path, vad_level=2, pitch=1.2, speed=1.1, voice="en-US-JennyNeural")
# If you want to play the sound:
# from playsound import playsound
# playsound(r"C:\Users\sabha\output.mp3")