#!/bin/bash
#
# Template: run your agent once on one H100 node of PACE-ICE, against an LLM served on the same node.
# Copy it to the root of your repository and edit the "Your settings" block and the last command.
#
# Build the image once, from the root of your repository, on a login node. Images are big, so keep the
# Apptainer cache and the image in a folder with space, not in your home directory. Pick that folder first
# (the path below is only an example: use a folder you can write to) and make sure it is set in this shell:
#   export SCRATCH_DIR=$HOME/scratch
#   mkdir -p "$SCRATCH_DIR"
#   export APPTAINER_CACHEDIR=$SCRATCH_DIR/cache APPTAINER_TMPDIR=$SCRATCH_DIR/tmp
#   mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
#   apptainer build --fakeroot "$SCRATCH_DIR/agent.sif" recipe/agent.def
#
# Run it, from the root of your repository:
#   sbatch run_agent.sbatch
#   sbatch --export=ALL,LLM_MODEL=openai/gpt-oss-20b run_agent.sbatch

#SBATCH --job-name=a4-agent
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=a4-agent-%j.out

set -euo pipefail

# ---- Your settings ---------------------------------------------------------------------------
SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch}"
IMAGE="${IMAGE:-$SCRATCH_DIR/agent.sif}"
LLM_MODEL="${LLM_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
RECORDINGS_DIR="${RECORDINGS_DIR:-my_recordings}"
OUTPUT_DIR="${OUTPUT_DIR:-run_output/${SLURM_JOB_ID}}"
# ----------------------------------------------------------------------------------------------

REPO_ROOT="${SLURM_SUBMIT_DIR}"
CPU_COUNT="${SLURM_CPUS_PER_TASK}"

# The professor's shared model cache. Nothing is downloaded at run time.
export HF_HOME=/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "node: $(hostname)  model: ${LLM_MODEL}  cpus: ${CPU_COUNT}"
nvidia-smi -L
module load apptainer 2>/dev/null || true

# ---- Serve the LLM -----------------------------------------------------------------------------
# vLLM comes from the shared A4 environment, outside your image.
CONDA_SH=/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
module load anaconda3 2>/dev/null || true
source "${CONDA_SH}"
conda activate /storage/ice1/0/4/lheck6/CS6604_ECE6256/envs/CAI_A4

# Ignore packages in your own ~/.local so that the shared environment is what runs.
export PYTHONNOUSERSITE=1

# Jobs that share a node share its network. A fixed port can silently reach somebody else's server,
# so each job takes a port from its own job id.
PORT=$((20000 + SLURM_JOB_ID % 20000))
VLLM_LOG="vllm-${SLURM_JOB_ID}.log"

python -m vllm.entrypoints.openai.api_server \
    --model "${LLM_MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.60 \
    --seed 42 \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'kill "${VLLM_PID}" 2>/dev/null || true' EXIT

# Loading the weights takes a few minutes.
for attempt in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "FATAL: vLLM stopped. The end of its log:"
        tail -n 40 "${VLLM_LOG}"
        exit 1
    fi
    sleep 5
done

# The server on the port must be yours and must serve your model.
if ! curl -s "http://127.0.0.1:${PORT}/v1/models" | grep -q "\"${LLM_MODEL}\""; then
    echo "FATAL: the server on port ${PORT} does not serve ${LLM_MODEL}"
    curl -s "http://127.0.0.1:${PORT}/v1/models" || true
    exit 1
fi
echo "LLM server ready on port ${PORT}"

# ---- Run the agent -------------------------------------------------------------------------------
# faster-whisper's GPU engine needs the CUDA libraries that pip installed inside the image.
# These folders are namespace packages, so their location comes from __path__.
CUDA_LIBS=$(apptainer exec --env PYTHONNOUSERSITE=1 "${IMAGE}" python3 - <<'PY'
import nvidia.cublas.lib as cublas
import nvidia.cudnn.lib as cudnn
print(":".join([list(cublas.__path__)[0], list(cudnn.__path__)[0]]))
PY
)

# PYTHONNOUSERSITE: Apptainer mounts your home directory, and packages in ~/.local would shadow the image's.
# OMP_NUM_THREADS covers some libraries only. Libraries like ONNX Runtime (Piper) and CTranslate2
# (faster-whisper) take their thread count from the node, not from this job, unless you pass it to them in
# code. Pass CPU_COUNT (or less) on, then check what it does to your stage timings.
apptainer exec --nv \
    --bind "${REPO_ROOT}:/work" \
    --bind "${HF_HOME}:${HF_HOME}:ro" \
    --env LD_LIBRARY_PATH="${CUDA_LIBS}" \
    --env PYTHONPATH=/work \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_OFFLINE=1 \
    --env PYTHONNOUSERSITE=1 \
    --env OMP_NUM_THREADS="${CPU_COUNT}" \
    "${IMAGE}" \
    python3 /work/your_agent.py \
    --recordings-directory "/work/${RECORDINGS_DIR}" \
    --output-directory "/work/${OUTPUT_DIR}" \
    --llm-model "${LLM_MODEL}" \
    --llm-base-url "http://127.0.0.1:${PORT}/v1" \
    --cpu-count "${CPU_COUNT}"
# ^ Replace your_agent.py and its arguments with your own. The model name and the server URL must reach
#   your agent as settings like these, never as constants in the code.
