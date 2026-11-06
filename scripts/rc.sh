#!/bin/bash
# Run a shell command on the Colab VM: scripts/rc.sh "<cmd>"
CMD="$1" python3 -c 'import os,json;print("import subprocess;r=subprocess.run(%s,shell=True,capture_output=True);print(r.stdout.decode(\"utf-8\",\"replace\")[-6000:]);print(r.stderr.decode(\"utf-8\",\"replace\")[-3000:])" % json.dumps(os.environ["CMD"]))' | colab --auth=adc exec -s ${SESSION:-isp} 2>&1 | grep -v -E "^\[colab\] (A new|Run .colab update|To silence)" | sed '/^$/N;/^\n$/D'
