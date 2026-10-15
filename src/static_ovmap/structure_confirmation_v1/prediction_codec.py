"""Lossless FP32 score transport: XOR adjacent rows, byte planes, gzip."""
import gzip
import json
from pathlib import Path
import numpy as np

def encode(path,values):
    a=np.asarray(values)
    if a.dtype!=np.float32:raise ValueError('Codec accepts exact FP32 scores only')
    a=np.ascontiguousarray(a.astype('<f4',copy=False));bits=a.view('<u4').reshape(-1,a.shape[-1]);delta=bits.copy()
    if len(bits)>1:delta[1:]=np.bitwise_xor(bits[1:],bits[:-1])
    planes=delta.view(np.uint8).reshape(-1,4).T.copy();header=json.dumps(dict(codec='FP32_XOR_PREVIOUS_ROW_BYTE_PLANES_GZIP_V1',shape=list(a.shape),dtype='<f4'),separators=(',',':')).encode()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with gzip.GzipFile(filename=str(temp),mode='wb',mtime=0,compresslevel=9) as stream:stream.write(len(header).to_bytes(4,'big'));stream.write(header);stream.write(planes.tobytes())
    temp.replace(path)

def decode(path):
    with gzip.open(path,'rb') as stream:
        n=int.from_bytes(stream.read(4),'big');header=json.loads(stream.read(n));raw=stream.read()
    if header['codec']!='FP32_XOR_PREVIOUS_ROW_BYTE_PLANES_GZIP_V1' or header['dtype']!='<f4':raise ValueError('Unknown score codec')
    shape=tuple(header['shape']);size=int(np.prod(shape))
    if len(raw)!=size*4:raise ValueError('Truncated score payload')
    delta=np.frombuffer(raw,np.uint8).reshape(4,-1).T.copy().reshape(-1).view('<u4').reshape(-1,shape[-1]);bits=np.bitwise_xor.accumulate(delta,axis=0)
    return bits.view('<f4').reshape(shape)
