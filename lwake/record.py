import logging
import numpy as np

from typing import Union
from pathlib import Path
import sounddevice as sd
import soundfile as sf
from silero_vad import get_speech_timestamps, load_silero_vad


_logger = logging.getLogger("local-wake")

def trim_silence_with_vad(audio:np.array, sample_rate:int)->np.array:
    """
        For the given audio array trims the silence 

    Args:
        audio (np.array): numpy array
        sample_rate (int): sample rate of the audio file 

    Returns:
        np.array: trimmed down version of audio
    """
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

def record(output:Union[str|Path], duration:int=3, trim_silence:bool=True, buffer_size:float=2.0, slide_size:float=0.25, stream:sd.InputStream = None):
    """Record audio from microphone and save it as a WAV file.

    This function captures audio for a fixed duration. If a pre-configured 
    `InputStream` is provided, it will be used for audio input; otherwise, 
    a new stream is created internally with default settings (16 kHz, mono, 
    float32). The recorded audio can optionally be trimmed to remove leading 
    and trailing silence using Voice Activity Detection (VAD).

    Args:
        output (Union[str, Path]): 
            File path (as string or pathlib.Path) where the recorded audio 
            will be saved in WAV format.
        duration (int, optional): 
            Recording duration in seconds. Must be a positive integer. 
            Defaults to 3.
        trim_silence (bool, optional): 
            If True, applies VAD-based silence trimming to remove non-speech 
            segments at the start and end of the recording. Defaults to True.
        buffer_size (float, optional): 
            Size of the internal audio buffer in seconds (used only if 
            `trim_silence=True`). Larger buffers allow more context for 
            silence detection but increase memory usage. Defaults to 2.0.
        slide_size (float, optional): 
            Step size (in seconds) for sliding window analysis during silence 
            trimming. Smaller values increase trimming precision but add 
            computational cost. Only used when `trim_silence=True`. 
            Defaults to 0.25.
        stream (sd.InputStream, optional): 
            An externally managed sounddevice InputStream. If provided, 
            the function reads audio from this stream instead of creating 
            a new one. The stream must already be started and configured 
            with compatible settings (sample rate = 16000 Hz, mono). 
            Defaults to None.

    
    Examples:
        >>> record("my_recording.wav", duration=5)
        >>> record(Path("audio/clip.wav"), duration=10, trim_silence=False)
        
        # Advanced: share stream with real-time processing
        >>> stream = sd.InputStream(samplerate=16000, channels=1, dtype='float32')
        >>> record("shared.wav", duration=3, stream=stream)
    """
    _logger.info(f"Recording for {duration} seconds...")
    
    if stream is None:
        stream = sd.InputStream(samplerate=16000, channels=1, dtype=np.float32)
    
    recorded_frames = 0 
    
    sample_rate = stream.samplerate
    total_samples = stream.samplerate* duration
    buffer_size_samples = int(buffer_size * sample_rate)
    slide_size_samples = int(slide_size * sample_rate)
    audio_buffer = np.zeros(buffer_size_samples, dtype=np.float32)
    
    _logger.info("Recording started")
    with stream:
        while recorded_frames< total_samples:
            data, overflowed = stream.read(slide_size_samples)
            if overflowed:
                _logger.warning("Audio buffer overflowed")
            
            chunk = data[:, 0]
            end_index = recorded_frames + chunk.shape[0]
            audio_buffer[recorded_frames:end_index] = chunk
            recorded_frames = end_index

    _logger.info("Recording Finished")
    if trim_silence:
        _logger.info("Trimming silence using VAD...")
        audio = trim_silence_with_vad(audio_buffer, stream.samplerate)
        
    sf.write(output, audio, samplerate=stream.samplerate)
    _logger.info(f"Saved to {output}")