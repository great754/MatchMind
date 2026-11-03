"""Bounded, pinned HTTP range ZIP reader. Never silently fetch an entire archive."""
import io,time,zipfile,requests
class RangeFile(io.RawIOBase):
 def __init__(self,url,size,budget=32000000):self.url=url;self.size=size;self.pos=0;self.bytes=0;self.budget=budget;self.session=requests.Session()
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if pos<0:raise ValueError('Negative position')
  self.pos=pos;return pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos)
  if n<=0:return b''
  if self.bytes+n>self.budget:raise ValueError('Remote read exceeds approved bounded byte budget')
  start=self.pos;end=start+n-1
  # Query-bust resolver caching; signed redirected URLs must not be stored in logs.
  r=self.session.get(self.url+('&' if '?' in self.url else '?')+f'range_start={start}&range_end={end}',headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=(15,60),stream=True)
  if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':r.close();raise ValueError(f'Range contract rejected: HTTP{r.status_code}, expected exact offset and archive size')
  data=r.raw.read(n+1);r.close()
  if len(data)!=n:raise ValueError('Range length mismatch')
  self.bytes+=n;self.pos+=n;return data

def open_remote(url,size,budget=32000000):
 f=RangeFile(url,size,budget);return zipfile.ZipFile(f),f

def member_bytes(url,size,entry,budget=2000000):
 """Read one member without unpacking paths or downloading adjacent frame data."""
 import struct,zlib,binascii
 offset=entry['offset'];n=min(size-offset,entry['compressed']+4096);f=RangeFile(url,size,budget);f.seek(offset);raw=f.read(n)
 if raw[:4]!=b'PK\x03\x04':raise ValueError('ZIP member header mismatch')
 fields=struct.unpack('<4s5H3I2H',raw[:30]);flags=fields[2];method=fields[3];name_length=fields[-2];extra_length=fields[-1]
 if flags&1:raise ValueError('Encrypted ZIP member is not authorized by this adapter')
 name=raw[30:30+name_length].decode('utf-8');start=30+name_length+extra_length
 if name!=entry['path'] or start+entry['compressed']>len(raw):raise ValueError('ZIP member name/length mismatch')
 payload=raw[start:start+entry['compressed']];data=zlib.decompress(payload,-15) if method==8 else payload if method==0 else None
 if data is None or len(data)!=entry['size'] or binascii.crc32(data)!=entry['crc']:raise ValueError('ZIP member CRC/size mismatch')
 return data,f.bytes
