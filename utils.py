# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:43:57 2021

@author: ted-17
"""

from scipy.signal import stft,resample_poly
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import read
from const import *

#-- function for feature extraction
def normalize8(X):
  '''
  normalize 0-1 unsigned 8 bit
  '''
  mn,mx = X.min(),X.max()
  mx -= mn
  X = ((X - mn)/mx)*255
  return X.astype(np.uint8)/255

def make_spectrogram_224(x,framesec=.025):
  f,t,S=stft(x,
             fs=FS,
             nperseg=int(framesec*FS),
             noverlap=int(framesec*FS*OVERLAPRATIO),
             nfft=FFTSIZE) #(257,226)
  S_mag=np.abs(S)
  S_log=np.log10(S_mag)
  S_log=S_log[:,::int(.025/framesec)]
  S_8bit=normalize8(S_log)[4:228,:]
  #-- スペクトログラムを224列ごとに区切る
  _S=S_8bit[:,:224*(S_8bit.shape[1]//224)] #余りを切り捨て 
  S_8bits=np.array_split(_S,_S.shape[1]//224,axis=1) #224列ごとに区切る

  return f[:224],t[:224],np.array(S_8bits)


def make_datasets(wavpaths):
    Ss=[]
    for wavfile in tqdm(wavpaths):    
      #-- read wavfile
      fs,x=read(wavfile)
      if fs!=FS:x=resample_poly(x,FS,fs)
      _,_,S_25ms=make_spectrogram_224(x,framesec=FRAMESEC25ms)
      if S_25ms.shape[1:]!=(224,224):
          print('skip:',S_25ms.shape)
          continue
      for i in range(S_25ms.shape[0]):
          _S=S_25ms[i,...]
          S=np.dstack((_S,_S,_S)) #3ch
          Ss.append(S)
      X=np.array(Ss)
    return X