# Kaggle kernel: prepare SIDD from mounted datasets and train one ISP model.
import glob, os, subprocess, sys, shutil
MODEL = "__MODEL__"; EXTRA = "__EXTRA__"
def find(pat):
    r = glob.glob(pat, recursive=True); print(pat, r[:2], flush=True); return r[0]
code_tgz = find("/kaggle/input/**/isp_code.tgz")
os.makedirs("/kaggle/tmp/code", exist_ok=True)
subprocess.run(f"tar xzf {code_tgz} -C /kaggle/tmp/code", shell=True, check=True)
raw = os.path.dirname(find("/kaggle/input/**/SIDD_Medium_Raw/Data/0001_001_S6_00100_00060_3200_L"))
srgb = os.path.dirname(find("/kaggle/input/**/SIDD_Small_sRGB_Only/Data/0001_001_S6_00100_00060_3200_L"))
subprocess.run("pip install -q h5py 2>&1 | tail -1; nvidia-smi --query-gpu=name --format=csv; nproc; df -h /kaggle/tmp | tail -1", shell=True)
os.chdir("/kaggle/tmp/code")
subprocess.run(f"python scripts/prep_sidd.py --raw {raw} --srgb {srgb} --out /kaggle/tmp/prep --workers 4 | tail -3", shell=True, check=True)
teacher = glob.glob("/kaggle/input/**/hybrid_teacher.pt", recursive=True)
extra = EXTRA.replace("TEACHER", teacher[0] if teacher else "")
init = glob.glob("/kaggle/input/**/hybrid_student_init.pt", recursive=True)
extra = extra.replace("INIT", init[0] if init else "")
cmd = f"python scripts/train.py --model {MODEL} --data /kaggle/tmp/prep --out /kaggle/working/{MODEL} --workers 4 {extra}"
print(cmd, flush=True)
subprocess.run(cmd, shell=True, check=True)
