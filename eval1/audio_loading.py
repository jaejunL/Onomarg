from __future__ import annotations
from pathlib import Path
import librosa, numpy as np
MAX_SAMPLES=320000
def load_audio(row:dict,*,exact_length:bool=False)->np.ndarray:
 path=Path(row['audio_path'])
 if not path.is_file():raise FileNotFoundError(path)
 if path.suffix.lower()=='.raw':
  audio=np.fromfile(path,dtype='<i2').astype(np.float32)/32768.0
  audio=librosa.resample(audio,orig_sr=16000,target_sr=32000,res_type='soxr_hq')
 else:
  audio,_=librosa.load(path,sr=32000,mono=True)
 audio=np.asarray(audio,dtype=np.float32)[:MAX_SAMPLES]
 if exact_length and audio.size<MAX_SAMPLES:audio=np.pad(audio,(0,MAX_SAMPLES-audio.size))
 return audio
