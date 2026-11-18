#!/usr/bin/env bash
set -euo pipefail

batch4=/home/lp531/smolvla_training/z1_left_wrist_no_phase_all_batch4_lr_restart_20260928_004609/checkpoints/075605/pretrained_model
batch8=/home/lp531/smolvla_training/z1_left_wrist_no_phase_all_batch8_lr_restart_20260928_004609/checkpoints/037803/pretrained_model
bottle_recovery=/home/lp531/smolvla_training/smolvla_bottle_recovery_boundary_only_batch8_20260930/checkpoints/030000/pretrained_model
pi05=/home/lp531/smolvla_training/pi05_fine_grip_recovery_base_20261012/checkpoints/120000/pretrained_model
pi05_close=/home/lp531/smolvla_training/pi05_grip_stage_base_20261011/checkpoints/120000/pretrained_model
pi05_correction=/home/lp531/smolvla_training/pi05_gentle_correction_base_20261010/checkpoints/080000/pretrained_model
pi05_continuous=/home/lp531/smolvla_training/pi05_bottle_three_camera_20261006/best
pi05_legacy=/home/lp531/smolvla_training/pi05_bottle_lora_20261001/best
act=/home/lp531/smolvla_training/act_correction_chunk16_20261013/checkpoints/050000/pretrained_model
act_fine=/home/lp531/smolvla_training/act_fine_cached_chunk16_20261015/checkpoints/050000/pretrained_model
act_gentle=/home/lp531/smolvla_training/act_gentle_cached_chunk16_20261015/checkpoints/050000/pretrained_model
act_apple=/home/lp531/smolvla_training/act_apple_gentle_cached_chunk16_20261018/checkpoints/050000/pretrained_model
diffusion_cup=/home/lp531/smolvla_training/diffusion_cup_gentle_cached_h16_20261020/checkpoints/050000/pretrained_model_ema
weight_file=model.safetensors
training_dataset=/home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_physical_checked_v1
model_args=()

dry_run=false
model=''
args=()
while (( $# )); do
  case "$1" in
    --help|-h)
      printf '用法：%s [--model 1|2|8|9|10|11|12|13|diffusion-cup|act-apple|act|act-gentle|act-fine|pi05|pi05-close|pi05-correction|batch4|batch8|bottle-recovery|pi05-continuous|pi05-legacy] [--objects cup|apple|peach|beverage_bottle] [其他验证参数]\n' "$0"
      printf '终端运行时，未指定模型或物体将显示选择菜单。\n'
      printf '默认使用10 Hz动作、60 Hz插值；闭爪后的双指接触等待已关闭。\n'
      printf 'pi05 使用二值标签：预测 >= 0.5 闭爪目标0 rad，否则开爪目标-0.90 rad；必须启用二值映射。pi05-continuous/pi05-legacy 使用原角度阈值 -0.75 rad。\n'
      printf 'bottle-recovery 为瓶子纠偏模型（batch 8，30000 step），默认测试瓶子。\n'
      printf 'pi05 为新基座重训纠偏2倍＋闭爪细分 π0.5（211条：203训练、8验证，chunk15，Attention+FFN LoRA，120000微步），每次执行5帧，瓶子保持自由物理状态；场景需 external_camera_layout:=dual。\n'
      printf '编号1（--model 1 或 pi05）：新纠偏＋闭爪细分120k；编号2（--model 2 或 pi05-correction）：第一版纠偏采样80k；编号8（--model 8 或 pi05-close）：上一版闭爪120k。\n'
      printf '编号10（--model 10 或 act-gentle）：温和采样 ACT 50k，三相机，chunk16，默认每次执行8帧。\n'
      printf '编号11（--model 11 或 act-fine）：精细阶段采样 ACT 50k，三相机，chunk16，默认每次执行15帧、闭爪最短保持1秒。\n'
      printf '编号12（--model 12 或 act-apple）：苹果基础＋纠偏温和采样 ACT 50k，三相机，chunk16，默认执行8帧、10 Hz动作。\n'
      printf '编号13（--model 13 或 diffusion-cup）：咖啡杯 Diffusion Policy EMA 50k，三相机、两帧观测、默认执行15步、10 Hz动作、闭爪保持1秒。\n'
      printf '编号9（--model 9 或 act）：ACT 50k参数更新，三相机，chunk16，每次执行8帧，二值夹爪；--dry-run 只预览命令。\n'
      printf 'pi05-legacy 保留旧双相机 π0.5 模型用于对照。\n'
      exit 0
      ;;
    --dry-run) dry_run=true; shift ;;
    --model)
      if (( $# < 2 )); then echo '--model 需要 1（新纠偏＋闭爪）、2（旧纠偏）、8（旧闭爪）、9（旧ACT 50k）、10（温和采样ACT 50k）、11（精细阶段ACT 50k）、12（苹果ACT 50k）、13（咖啡杯Diffusion EMA）、diffusion-cup、act-apple、act、act-gentle、act-fine、batch4、batch8、bottle-recovery、pi05、pi05-correction、pi05-continuous 或 pi05-legacy。' >&2; exit 2; fi
      model=$2
      shift 2
      ;;
    --model=*) model=${1#*=}; shift ;;
    --checkpoint|--checkpoint=*) echo '请使用 --model 1|2|8|9|10|11|12|13|diffusion-cup|act-apple|act|act-gentle|act-fine|pi05|pi05-close|pi05-correction|batch4|batch8|bottle-recovery|pi05-continuous|pi05-legacy 选择模型。' >&2; exit 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
set -- "${args[@]}"

if [[ -z "$model" ]]; then
  if [[ ! -t 0 ]]; then
    echo '请在终端选择模型，或传入 --model 1|2|8|9|10|11|12|13|diffusion-cup|act-apple|act|act-gentle|act-fine|pi05|pi05-close|pi05-correction|batch4|batch8|bottle-recovery|pi05-continuous|pi05-legacy。' >&2
    exit 2
  fi
  printf '选择验证模型：\n1) 新纠偏＋闭爪细分 π0.5（120000微步）\n2) 第一版纠偏采样 π0.5（80000微步）\n3) batch 4 续训最终模型\n4) batch 8 续训最终模型\n5) 瓶子纠偏 SmolVLA（30000 step）\n6) 旧双相机 π0.5\n7) 上一版连续标签三相机 π0.5\n8) 上一版加强闭爪采样 π0.5（120000微步）\n9) ACT 原始＋纠偏（50000次参数更新）\n10) ACT 温和采样（50000次参数更新，20261015）\n11) ACT 精细阶段采样（50000次参数更新，首闭爪帧2倍）\n12) 苹果 ACT 基础＋纠偏温和采样（50000次更新，20261018）\n13) 咖啡杯 Diffusion Policy 温和采样 EMA（50000次更新，20261020）\n'
  read -r -p '输入 1-13: ' selection
  case "$selection" in
    1) model=pi05 ;;
    2) model=pi05-correction ;;
    3) model=batch4 ;;
    4) model=batch8 ;;
    5) model=bottle-recovery ;;
    6) model=pi05-legacy ;;
    7) model=pi05-continuous ;;
    8) model=pi05-close ;;
    9) model=act ;;
    10) model=act-gentle ;;
    11) model=act-fine ;;
    12) model=act-apple ;;
    13) model=diffusion-cup ;;
    *) echo '无效模型选择。' >&2; exit 2 ;;
  esac
fi

case "$model" in
  1) model=pi05 ;;
  2) model=pi05-correction ;;
  8) model=pi05-close ;;
  9) model=act ;;
  10) model=act-gentle ;;
  11) model=act-fine ;;
    12) model=act-apple ;;
    13) model=diffusion-cup ;;
esac

case "$model" in
  diffusion-cup|diffusion)
    model=diffusion-cup
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export FASTRTPS_DEFAULT_PROFILES_FILE=/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/fastdds_camera.xml
    export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
    checkpoint=$diffusion_cup
    training_dataset=/home/lp531/smolvla_data/z1_tabletop_raw_cup_standard_three_camera_20261019
    model_args=(--policy-type diffusion --target-xy-bounds 0.36 0.69 -0.27 0.074 --execute-steps 15 --binary-gripper --gripper-action-encoding binary --gripper-min-close-hold-s 1 --no-freeze-target-until-close --output /home/lp531/smolvla_eval/diffusion_cup_gentle_cached_h16_20261020_50000)
    ;;
  act|act-gentle|act-fine|act-apple)
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export FASTRTPS_DEFAULT_PROFILES_FILE=/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/fastdds_camera.xml
    export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
    checkpoint=$act
    act_steps=8
    act_output=/home/lp531/smolvla_eval/act_correction_chunk16_20261013_50000
    if [[ "$model" == act-gentle ]]; then
      checkpoint=$act_gentle
      act_output=/home/lp531/smolvla_eval/act_gentle_cached_chunk16_20261015_50000
    fi
    if [[ "$model" == act-fine ]]; then
      checkpoint=$act_fine
      act_steps=15
      act_output=/home/lp531/smolvla_eval/act_fine_cached_chunk16_20261015_50000
    fi
    training_dataset=/home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_three_camera_v1
    model_args=(--policy-type act --target-xy-bounds 0.35 0.70 -0.27 0.16 --execute-steps "$act_steps" --binary-gripper --gripper-action-encoding binary --no-freeze-target-until-close --output "$act_output")
    if [[ "$model" == act-apple ]]; then
      checkpoint=$act_apple
      training_dataset=/home/lp531/smolvla_data/z1_tabletop_raw_apple_preplanned_three_camera_20261016
      model_args=(--policy-type act --target-xy-bounds 0.36 0.624 -0.27 0.16 --execute-steps "$act_steps" --binary-gripper --gripper-action-encoding binary --no-freeze-target-until-close --output /home/lp531/smolvla_eval/act_apple_gentle_cached_chunk16_20261018_50000)
    fi
    if [[ "$model" == act-fine ]]; then
      model_args+=(--gripper-min-close-hold-s 1)
    fi
    ;;
  batch4) checkpoint=$batch4 ;;
  batch8) checkpoint=$batch8 ;;
  bottle-recovery)
    checkpoint=$bottle_recovery
    training_dataset=/home/lp531/smolvla_data/z1_tabletop_bottle_recovery_boundary_only_prepared_v1
    model_args=(--output /home/lp531/smolvla_eval/bottle_recovery_boundary_only_batch8_30000)
    ;;
  pi05|pi05-close|pi05-correction|pi05-continuous)
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export FASTRTPS_DEFAULT_PROFILES_FILE=/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/fastdds_camera.xml
    export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
    checkpoint=$pi05
    if [[ "$model" == pi05-close ]]; then checkpoint=$pi05_close; fi
    if [[ "$model" == pi05-correction ]]; then checkpoint=$pi05_correction; fi
    if [[ "$model" == pi05-continuous ]]; then checkpoint=$pi05_continuous; fi
    weight_file=adapter_model.safetensors
    training_dataset=/home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_three_camera_v1
    model_args=(--policy-type pi05 --execute-steps 5 --binary-gripper --no-freeze-target-until-close --output /home/lp531/smolvla_eval/pi05_bottle_three_camera_best)
    if [[ "$model" == pi05 ]]; then
      model_args=(--policy-type pi05 --execute-steps 5 --binary-gripper --gripper-action-encoding binary --no-freeze-target-until-close --output /home/lp531/smolvla_eval/pi05_fine_grip_recovery_base_20261012_120000)
    fi
    if [[ "$model" == pi05-close ]]; then
      model_args=(--policy-type pi05 --execute-steps 5 --binary-gripper --gripper-action-encoding binary --no-freeze-target-until-close --output /home/lp531/smolvla_eval/pi05_grip_stage_base_20261011_120000)
    fi
    if [[ "$model" == pi05-correction ]]; then
      model_args=(--policy-type pi05 --execute-steps 5 --binary-gripper --gripper-action-encoding binary --no-freeze-target-until-close --output /home/lp531/smolvla_eval/pi05_gentle_correction_base_20261010)
    fi
    ;;
  pi05-legacy)
    checkpoint=$pi05_legacy
    weight_file=adapter_model.safetensors
    training_dataset=/home/lp531/smolvla_data/z1_tabletop_bottle_recovery_boundary_only_prepared_v1
    model_args=(--policy-type pi05 --execute-steps 5 --binary-gripper --no-freeze-target-until-close --output /home/lp531/smolvla_eval/pi05_bottle_lora_best)
    ;;
  *) echo '无效模型；请选择 1（新纠偏＋闭爪）、2（旧纠偏）、8（旧闭爪）、9（旧ACT 50k）、10（温和采样ACT 50k）、11（精细阶段ACT 50k）、12（苹果ACT 50k）、13（咖啡杯Diffusion EMA）、diffusion-cup、act-apple、act、act-gentle、act-fine、batch4、batch8、bottle-recovery、pi05、pi05-correction、pi05-continuous 或 pi05-legacy。' >&2; exit 2 ;;
esac
if [[ ! -s "$checkpoint/$weight_file" || ! -s "$checkpoint/config.json" ]]; then
  echo "找不到完整的 $model checkpoint：$checkpoint" >&2
  exit 1
fi

choose_object=true
for arg in "$@"; do
  case "$arg" in
    --objects|--objects=*|--help|-h) choose_object=false ;;
  esac
done

if $choose_object && [[ "$model" == diffusion-cup ]]; then
  set -- --objects cup "$@"
  choose_object=false
fi

if $choose_object && [[ "$model" == act-apple ]]; then
  set -- --objects apple "$@"
  choose_object=false
fi

if $choose_object && [[ "$model" == act || "$model" == act-gentle || "$model" == act-fine || "$model" == bottle-recovery || "$model" == pi05 || "$model" == pi05-close || "$model" == pi05-correction || "$model" == pi05-continuous || "$model" == pi05-legacy ]]; then
  set -- --objects beverage_bottle "$@"
  choose_object=false
fi

if $choose_object; then
  if [[ ! -t 0 ]]; then
    echo '请在终端选择物体，或传入 --objects cup|apple|peach|beverage_bottle。' >&2
    exit 2
  fi
  printf '选择验证物体：1) 杯子  2) 苹果  3) 桃子  4) 瓶子  5) 全部\n'
  read -r -p '输入 1-5: ' selection
  case "$selection" in
    1) set -- --objects cup "$@" ;;
    2) set -- --objects apple "$@" ;;
    3) set -- --objects peach "$@" ;;
    4) set -- --objects beverage_bottle "$@" ;;
    5) set -- --objects cup apple peach beverage_bottle "$@" ;;
    *) echo '无效选择。' >&2; exit 2 ;;
  esac
fi

if [[ "$model" == diffusion-cup || "$model" == act-apple || "$model" == act || "$model" == act-gentle || "$model" == act-fine || "$model" == pi05 || "$model" == pi05-close || "$model" == pi05-correction || "$model" == pi05-continuous ]]; then
  echo '三相机验证：场景须使用 external_camera_layout:=dual，并发布 wrist/global/global_right。'
fi
echo "验证 $model checkpoint：$checkpoint"
command=(/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_sim_eval.sh \
  --checkpoint "$checkpoint" \
  --wrist-rgb-topic /left_wrist_camera/color/image_raw \
  --training-dataset-root "$training_dataset" \
  --max-training-target-distance-m 0.03 \
  --reject-blank-cameras \
  --episodes-per-object 1 \
  --rate 10 --execution-rate 60 \
  --prelift-contact-stable-s 0 --prelift-contact-timeout-s 2.0 \
  "${model_args[@]}" \
  "$@")
if $dry_run; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
