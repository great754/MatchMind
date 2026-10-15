"""Build separately, then verify an ephemeral container using real HTTP."""
import json
from pathlib import Path
import subprocess
import time
import requests
from smoke_service import smoke

def main():
    inspect=json.loads(subprocess.check_output(['docker','image','inspect','matchmind-service'],text=True))[0]
    container=subprocess.check_output(['docker','run','--rm','-d','-p','127.0.0.1:8767:8000','-v',str(Path('outputs').resolve())+':/app/outputs:ro','matchmind-service'],text=True).strip()
    start=time.perf_counter()
    try:
        session=requests.Session();session.trust_env=False
        for _ in range(200):
            try:
                if session.get('http://127.0.0.1:8767/ready',timeout=1).status_code==200:break
            except requests.RequestException:pass
            time.sleep(.1)
        ready=time.perf_counter()-start
        result=smoke('http://127.0.0.1:8767','118575_clip')
        print(json.dumps(dict(image='matchmind-service',size_bytes=inspect['Size'],user=inspect['Config']['User'],exposed_ports=inspect['Config']['ExposedPorts'],startup_seconds=ready,smoke=result),indent=2))
    finally:subprocess.run(['docker','stop',container],check=True,stdout=subprocess.DEVNULL)
if __name__=='__main__':main()
