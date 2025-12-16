import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np


@dataclass
class StreamConfig:
    samplerate: int = 16000
    blocksize: Optional[int] = None
    device: Optional[int | str] = None
    channels: int = 1
    dtype: str = "float32"
    latency: Optional[str | float] = "low"

    extra_settings: Optional[Any] = None
    callback: Optional[Callable] = None
    finished_callback: Optional[Callable] = None

    clip_off: Optional[bool] = None
    dither_off: Optional[bool] = None
    never_drop_input: Optional[bool] = None
    prime_output_buffers_using_stream_callback: Optional[bool] = None


@dataclass
class RecordConfig:
    frames: Optional[int] = 48000  # 3 (sec) * 16000 (sample rate)
    samplerate: Optional[int] = 16000
    channels: Optional[int] = 1
    dtype: Optional[Union[str, np]] = np.float32
    out: Optional[Any] = None
    mapping: Optional[list[int]] = None
    blocking: bool = False

    def calculate_frame(self, duration: int):
        """Function used to calculate the frames given the duration"""
        self.frames = int(round(duration * self.samplerate))
        return self.frames

    
    

