#!/bin/bash
# GSM8K GRPO 训练脚本 - 4卡 H20 (2推理 + 2训练)
# 使用方法: bash examples/math/run_gsm8k_4gpu_h20.sh
#
# 环境变量:
#   EXPERIMENT_NAME - 实验名称 (默认: gsm8k-grpo-4gpu)
#   TRIAL_NAME - 试验名称 (默认: trial-时间戳)
#   MODEL_PATH - 模型路径 (默认: /root/paddlejob/workspace/long/Qwen3-8B-FP8)
#   WANDB_API_KEY - WandB API Key (必须设置，否则 WandB 无法工作)
#   BATCH_SIZE - 批次大小 (默认: 64)

set -e

# 切换到 AReaL 目录
cd "$(dirname "$0")/../.."

# 设置实验名称和试验名称
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"gsm8k-grpo-4gpu"}
TRIAL_NAME=${TRIAL_NAME:-"trial-$(date +%Y%m%d-%H%M%S)"}

# 模型路径（可通过环境变量覆盖）
MODEL_PATH=${MODEL_PATH:-"/root/paddlejob/workspace/long/Qwen3-8B-FP8"}

# 批次大小
BATCH_SIZE=${BATCH_SIZE:-64}

# WandB 配置
export WANDB_PROJECT=${WANDB_PROJECT:-"gsm8k-grpo"}

echo "=========================================="
echo "GSM8K GRPO Training on 4x H20 GPUs (vLLM)"
echo "=========================================="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Trial: ${TRIAL_NAME}"
echo "Model: ${MODEL_PATH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Inference Backend: vLLM"
echo "WandB Project: ${WANDB_PROJECT}"
echo "=========================================="

# ========== 前置检查 ==========

# 检查 GPU 数量
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
echo "[CHECK] 检测到 ${GPU_COUNT} 个 GPU"
if [ "$GPU_COUNT" -lt 4 ]; then
    echo "[WARNING] 需要至少 4 个 GPU，当前只有 ${GPU_COUNT} 个"
    echo "         如果继续，可能会失败"
fi

# 检查模型路径
if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] 模型路径不存在: ${MODEL_PATH}"
    echo "        请设置正确的 MODEL_PATH 环境变量"
    exit 1
fi
echo "[CHECK] 模型路径存在: ${MODEL_PATH}"

# 检查是否是 FP8 模型 (检查 config.json 中是否有 quantization_config)
if [ -f "${MODEL_PATH}/config.json" ]; then
    if grep -q "fp8" "${MODEL_PATH}/config.json" 2>/dev/null; then
        echo "[WARNING] 检测到 FP8 量化模型"
        echo "          FP8 模型在 RL 训练时可能有精度问题"
        echo "          建议使用非 FP8 模型: Qwen/Qwen3-8B 或 Qwen/Qwen2.5-7B-Instruct"
    fi
fi

# 检查 WandB API Key
if [ -z "${WANDB_API_KEY}" ]; then
    echo "[WARNING] WANDB_API_KEY 未设置"
    echo "          WandB 将无法上传日志到云端"
    echo "          设置方法: export WANDB_API_KEY=your_api_key"
    echo ""
    read -p "是否继续运行（WandB 将以离线模式运行）? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    # 切换到离线模式
    WANDB_MODE="offline"
else
    WANDB_MODE="online"
    echo "[CHECK] WandB API Key 已设置"
fi

# 检查必要的目录
mkdir -p /tmp/areal/experiments
mkdir -p /tmp/areal/name_resolve
echo "[CHECK] 必要目录已创建"

# 禁用代理（内部服务通信）
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

# NCCL 配置（避免与推理引擎的 NCCL 冲突）
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0

echo "=========================================="
echo "开始训练..."
echo "=========================================="

# 运行训练 (使用 vLLM 推理后端)
python3 examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo_4gpu_h20.yaml \
    experiment_name="${EXPERIMENT_NAME}" \
    trial_name="${TRIAL_NAME}" \
    actor.path="${MODEL_PATH}" \
    ref.path="${MODEL_PATH}" \
    vllm.model="${MODEL_PATH}" \
    train_dataset.batch_size="${BATCH_SIZE}" \
    valid_dataset.batch_size="${BATCH_SIZE}" \
    stats_logger.wandb.mode="${WANDB_MODE}" \
    stats_logger.wandb.name="${EXPERIMENT_NAME}-${TRIAL_NAME}" \
    "$@"

echo "=========================================="
echo "训练完成!"
echo "日志目录: /tmp/areal/experiments/${EXPERIMENT_NAME}/${TRIAL_NAME}"
echo "=========================================="