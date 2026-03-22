# GCP VM Setup Guide — RL Dissertation Training

Run all 4 Atari training experiments on a GCP T4 GPU VM.
Estimated cost: **$3–8 total** using Spot pricing.
Estimated wall time: **18–25 hours** (all 4 runs sequentially).

---

## Quick Reference

| Run | Algorithm | Game | Steps | ~Time (T4) |
|-----|-----------|------|-------|------------|
| 1 | DQN | Pong | 2M | ~3–4 hours |
| 2 | DQN | Breakout | 5M | ~7–9 hours |
| 3 | DDQN | Pong | 2M | ~3–4 hours |
| 4 | DDQN | Breakout | 5M | ~7–9 hours |

---

## Step 1 — One-Time GCP Setup

### 1a. Create a project and enable billing

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Click the project dropdown (top left) → **New Project**
   - Name: `rl-dissertation` → **Create**
3. Go to **Billing** (left sidebar) → link your billing account
   - Your $300 free credits will be used — you will not be charged unless you exceed them

### 1b. Request GPU quota

1. Go to **IAM & Admin → Quotas & System Limits** (left sidebar)
2. In the filter box type: `NVIDIA T4 GPUs`
3. Find the row for region **`us-central1`** (or `us-east1` as a backup)
4. Tick the checkbox → click **Edit Quotas** at the top
5. Set new value to **1** → fill in your name/email → **Submit Request**
6. Approval is usually instant or within a few hours — you'll get an email

> Skip this step and proceed — quota is sometimes already set to 1 by default. You'll only hit an error when creating the VM if it's not.

---

## Step 2 — Create the VM (GCP Console)

1. Go to **Compute Engine → VM instances** (left sidebar)
2. Click **Create Instance** at the top

Fill in the form exactly as follows:

**Name & Region**
- Name: `rl-trainer`
- Region: `us-central1` / Zone: `us-central1-a`

**Machine configuration**
- Series: **N1**
- Machine type: **n1-standard-4** (4 vCPUs, 15 GB memory)

**GPU**
- Click **Add a GPU**
- GPU type: **NVIDIA T4**
- Number of GPUs: **1**

**Availability policies** (this is what makes it cheap)
- VM provisioning model: **Spot**
- On VM termination: **Stop** (NOT Delete — this keeps your disk safe)

**Boot disk**
- Click **Change**
- Go to the **Custom images** tab, then switch to **Public images**
- Operating system: **Deep Learning on Linux**
- Version: **Deep Learning VM with CUDA 12.1 M126** (or the latest M-version available)
- Boot disk type: **Balanced persistent disk**
- Size: **50 GB**
- Click **Select**

**Metadata** (scroll down to Advanced Options → Management → Metadata)
- Click **Add item**
- Key: `install-nvidia-driver` / Value: `True`

3. Click **Create**

**Cost:** ~$0.11–0.16/hr (Spot) vs ~$0.45/hr (Standard). Total ~$3–8 for the full project.

> **If you get a quota error on Spot:** Change VM provisioning model to **Standard** instead. Cost goes to ~$8–15 total, still well within $300.
>
> **If `us-central1` has no T4 capacity:** Try zone `us-east1-d` or `us-west1-b` instead.

---

## Step 3 — SSH Into the VM

1. Go to **Compute Engine → VM instances**
2. Wait until the VM status shows a green tick (ready) — takes about 1–2 minutes
3. Click the **SSH** button in the Connect column — a browser terminal opens automatically

Wait ~2–3 minutes on first boot for the NVIDIA driver to finish installing. If you see a driver installation message, wait for it to complete before running any Python.

Verify the GPU is working:
```bash
nvidia-smi
```
You should see the T4 listed with ~16GB memory and a driver version.

---

## Step 4 — Clone the Project onto the VM

In the browser SSH terminal:

```bash
cd ~
git clone https://github.com/vinoltauro/rl-project.git rl_project
cd rl_project
```

That's it — all code and configs are now on the VM.

> **To pull updates later** (if you push fixes from your PC):
> ```bash
> cd ~/rl_project
> git pull
> ```

---

## Step 5 — Install Dependencies

```bash
cd ~/rl_project

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install gymnasium[atari] ale-py numpy pandas matplotlib seaborn \
            scikit-learn tensorboard pyyaml tqdm opencv-python AutoROM

# Install Atari ROMs
AutoROM --accept-license

# Quick sanity check
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA: True
GPU: Tesla T4
```

---

## Step 6 — Run Training in tmux

**tmux lets training continue even if you close your SSH session.**

```bash
# Install tmux (if not available)
sudo apt-get install -y tmux

# Start a named session
tmux new -s training

# Inside tmux — start all 4 runs
cd ~/rl_project
python run_all.py --training_only
```

**Detach from tmux** (training keeps running after you close SSH):
```
Ctrl+B, then D
```

**Reattach later** (SSH back in and run):
```bash
tmux attach -t training
```

### If the VM gets preempted (Spot VMs only)

Spot VMs can be stopped by GCP with no warning. Your data on disk is safe. Just:
1. Go to **Compute Engine → VM instances** in the console
2. Tick the `rl-trainer` checkbox → click **Start / Resume** at the top
3. Once it's running, click **SSH** again
4. Re-run — `run_all.py` will automatically detect the latest checkpoint and resume:
   ```bash
   tmux new -s training
   cd ~/rl_project
   python run_all.py --training_only
   ```
   It skips completed runs and auto-resumes any interrupted run from the last checkpoint.

---

## Step 7 — Monitor Progress

**Open a second terminal while training runs:**
1. Go to **Compute Engine → VM instances** in the console
2. Click **SSH** again — opens a second browser terminal
3. Reattach to tmux to see live output:
```bash
tmux attach -t training
```

**Check GPU utilisation:**
```bash
watch -n 5 nvidia-smi
```
You should see ~90%+ GPU utilisation during training.

**Check log files:**
```bash
ls ~/rl_project/results/logs/
tail -f ~/rl_project/results/logs/*.csv
```

**Check checkpoints:**
```bash
ls -lh ~/rl_project/results/checkpoints/
```

---

## Step 8 — Run Analysis After Training

Once all 4 runs are complete:
```bash
cd ~/rl_project
python run_all.py --analysis_only
```

This generates all 17 figures in `results/plots/`.

---

## Step 9 — Download Results to Your PC

In the browser SSH terminal, zip up the results first:
```bash
cd ~
zip -r rl_results.zip rl_project/results/
```

Then download:
1. In the browser SSH terminal, click the **gear icon** (top right) → **Download file**
2. Enter the file path: `/home/YOUR_USERNAME/rl_results.zip`
3. Click **Download** — saves to your PC's Downloads folder

> To find your username run `echo $HOME` in the terminal.

**Just the plots** (much smaller — ~50MB vs ~2GB for everything):
```bash
zip -r rl_plots.zip rl_project/results/plots/
```
Then download `rl_plots.zip` the same way.

---

## Step 10 — STOP THE VM (Important!)

**Always stop the VM when you're done — billing continues while it's running.**

1. Go to **Compute Engine → VM instances**
2. Tick the `rl-trainer` checkbox
3. Click **Stop** at the top → confirm

To delete the VM entirely (frees up disk storage cost too — ~$1/month for 50GB):
- Instead of **Stop**, click **Delete** → confirm

> Stopped VMs do not incur compute charges. Only the persistent disk costs continue (~$1/month for 50GB) — negligible.

---

## Cost Summary

| Item | Rate | Estimated Use | Cost |
|------|------|---------------|------|
| T4 Spot VM (n1-standard-4 + T4) | ~$0.13/hr | ~20–25 hours | ~$3–4 |
| Boot disk (50GB pd-balanced) | ~$0.02/hr | ~25 hours | ~$0.50 |
| Egress (downloading results) | ~$0.08/GB | ~2GB results | ~$0.16 |
| **Total** | | | **~$4–5** |

**Your $300 credits covers roughly 2,000+ hours of this VM** — you have enormous headroom.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvidia-smi: command not found` after SSH | Wait 3–5 minutes, driver is still installing. Re-login. |
| `CUDA not available` in Python | Run `sudo /opt/deeplearning/install-driver.sh` and re-login |
| Spot VM keeps getting preempted | In the console, try a different zone: edit the VM → change zone to `us-east1-d` or `us-west1-b`. Or change provisioning model to **Standard**. |
| Quota error when creating VM | Go to IAM & Admin → Quotas → request NVIDIA T4 quota = 1 for the zone you're using (Step 1b) |
| `AutoROM` fails | Try `pip install autorom` then `python -m autorom.run --accept-license` |
| Out of disk space | Console → Compute Engine → Disks → click `rl-trainer` → Edit → increase size to 100GB |
| SSH button won't connect | VM may still be booting — wait 60 seconds and click SSH again |

---

## Alternative: Google Colab (No VM Needed)

If GCP quota approval takes too long, use Colab with Google Drive for persistence.

```python
# Cell 1 — Mount Drive (run this first — keeps data across sessions)
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — Install dependencies
!pip install gymnasium[atari] ale-py numpy pandas matplotlib seaborn \
             scikit-learn tensorboard pyyaml tqdm opencv-python AutoROM -q
!AutoROM --accept-license

# Cell 3 — Upload and extract project
# Upload rl_project.tar.gz via the Files panel (left sidebar), then:
import tarfile, os
with tarfile.open('rl_project.tar.gz') as t:
    t.extractall('/content/')
os.chdir('/content/rl_project')

# Cell 4 — Copy any existing results from Drive (for resuming)
import shutil
drive_results = '/content/drive/MyDrive/rl_results'
if os.path.exists(drive_results):
    shutil.copytree(drive_results, 'results', dirs_exist_ok=True)
    print("Restored results from Drive")

# Cell 5 — Run training (one run at a time to avoid timeout)
!python run_all.py --runs 1 --training_only

# Cell 6 — Save results to Drive immediately after each run
shutil.copytree('results', '/content/drive/MyDrive/rl_results', dirs_exist_ok=True)
print("Results saved to Drive")
```

**Colab workflow:**
1. Run Cells 1–4 at the start of every session
2. Run one training run at a time (Pong ~3–4h, Breakout ~7–9h — may need multiple sessions)
3. Save to Drive after EVERY run (Cell 6)
4. If session dies: start new session, restore from Drive (Cell 4), re-run — `run_all.py` auto-resumes

**Colab Pro** ($10/month) gives longer sessions and priority GPU access — worth it if free sessions keep timing out.
