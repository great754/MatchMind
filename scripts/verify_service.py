"""Start a real local server, execute HTTP smoke checks, and stop it."""
import json
from pathlib import Path
import subprocess
import sys
import time
import requests
from smoke_service import smoke

def main():
    with Path('/tmp/matchmind_http_smoke.log').open('w') as log:
        server=subprocess.Popen([sys.executable,'-m','uvicorn','src.serving.app:from_environment','--factory','--host','127.0.0.1','--port','8766','--no-access-log'],stdout=log,stderr=log)
        try:
            session=requests.Session();session.trust_env=False
            for _ in range(100):
                if server.poll() is not None:raise RuntimeError('Service exited; inspect /tmp/matchmind_http_smoke.log')
                try:
                    if session.get('http://127.0.0.1:8766/ready',timeout=1).status_code==200:break
                except requests.RequestException:pass
                time.sleep(.1)
            print(json.dumps(smoke('http://127.0.0.1:8766','118575_clip'),indent=2))
        finally:
            server.terminate();server.wait(timeout=10)
if __name__=='__main__':main()
