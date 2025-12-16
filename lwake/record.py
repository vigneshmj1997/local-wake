import logging
from dataclasses import asdict

import sounddevice as sd
import soundfile as sf
from silero_vad import get_speech_timestamps, load_silero_vad

from lwake.config import RecordConfig

_logger = logging.getLogger("local-wake")

def trim_silence_with_vad(audio, sample_rate):
    _logger.info("Loading Silero VAD model...")
    model = load_silero_vad()

    speech_timestamps = get_speech_timestamps(
        audio[:, 0], model,
        sampling_rate=sample_rate,
    )
    
    if not speech_timestamps:
        _logger.warning("No speech detected in audio")
        return audio
    
    start_sample = speech_timestamps[0]['start']
    end_sample = speech_timestamps[-1]['end']
    
    _logger.info(f"Trimmed audio to [{start_sample/sample_rate:.2f}s, {end_sample/sample_rate:.2f}s]")
    return audio[start_sample:end_sample]

def record(output, trim_silence=True, record_config = RecordConfig()):
    """Record audio and save as WAV file"""
    _logger.info(f"Recording for {record_config.frames/record_config.samplerate} seconds...")
    
    audio = sd.rec(**asdict(record_config))
    sd.wait()

    if trim_silence:
        _logger.info("Trimming silence using VAD...")
        audio = trim_silence_with_vad(audio, record_config.samplerate)
        
    sf.write(output, audio, samplerate=record_config.samplerate)
    _logger.info(f"Saved to {output}")