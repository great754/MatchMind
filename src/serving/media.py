"""Private Azure clip delivery through the authenticated API, with HTTP seeking."""
import logging
import re
from fastapi import HTTPException
from fastapi.responses import Response,StreamingResponse


class AzureClipMedia:
    def __init__(self,client):
        self.client=client
        self.metadata={}

    def properties(self,match_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',match_id):
            return None
        if match_id not in self.metadata:
            try:
                self.metadata[match_id]=self.client.get_blob_client(f'mot/clips/{match_id}.mp4').get_blob_properties()
            except Exception:
                logging.getLogger('matchmind.requests').warning('Azure video unavailable for %s',match_id)
                self.metadata[match_id]=None
        return self.metadata[match_id]

    def available(self,match_id):
        properties=self.properties(match_id)
        return properties is not None and properties.size>0

    def response(self,match_id,request):
        properties=self.properties(match_id)
        if properties is None or properties.size<=0:
            raise HTTPException(404,'Video unavailable')
        size=properties.size
        headers={'Accept-Ranges':'bytes','Content-Type':'video/mp4','ETag':properties.etag}
        value=request.headers.get('range')
        # If-Range mismatch requests the complete current representation.
        if request.headers.get('if-range') not in (None,properties.etag):value=None
        start=0;end=size-1;status=200
        if value:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
            if not match or not any(match.groups()):
                return Response(status_code=416,headers={**headers,'Content-Range':f'bytes */{size}'})
            first,last=match.groups()
            if first:
                start=int(first);end=min(int(last),size-1) if last else size-1
            else:
                length=int(last);start=max(0,size-length)
                if length==0:start=size
            if start>=size or start>end:
                return Response(status_code=416,headers={**headers,'Content-Range':f'bytes */{size}'})
            status=206;headers['Content-Range']=f'bytes {start}-{end}/{size}'
        length=end-start+1;headers['Content-Length']=str(length)
        if request.method=='HEAD':return Response(status_code=status,headers=headers)
        try:
            blob=self.client.get_blob_client(f'mot/clips/{match_id}.mp4')
            # An ETag condition prevents mixing bytes from different revisions.
            from azure.core import MatchConditions
            download=blob.download_blob(offset=start,length=length,etag=properties.etag,match_condition=MatchConditions.IfNotModified)
        except Exception:
            raise HTTPException(503,'Azure video download unavailable') from None
        return StreamingResponse(download.chunks(),status_code=status,headers=headers)
