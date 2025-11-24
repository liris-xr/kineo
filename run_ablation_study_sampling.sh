#!/bin/bash

declare -A EGOHUMANS_SEQUENCES=(
    ["tagging"]="tagging_001 tagging_002 tagging_003 tagging_004 tagging_005 tagging_006 tagging_007 tagging_008 tagging_009 tagging_010 tagging_011 tagging_012 tagging_013 tagging_014"
    ["legoassemble"]="legoassemble_001 legoassemble_002 legoassemble_003 legoassemble_004 legoassemble_005 legoassemble_006"
    ["fencing"]="fencing_001 fencing_002 fencing_003 fencing_004 fencing_005 fencing_006 fencing_007 fencing_008 fencing_009 fencing_010 fencing_011 fencing_012 fencing_013 fencing_014"
    ["basketball"]="basketball_001 basketball_002 basketball_003 basketball_004 basketball_005 basketball_006 basketball_007 basketball_008 basketball_009 basketball_011 basketball_012 basketball_013 basketball_014"
    ["volleyball"]="volleyball_001 volleyball_002 volleyball_003 volleyball_004 volleyball_005 volleyball_006 volleyball_007 volleyball_008 volleyball_009 volleyball_010 volleyball_011"
    ["badminton"]="badminton_001 badminton_002 badminton_003 badminton_004 badminton_005 badminton_006 badminton_007 badminton_008 badminton_009 badminton_010 badminton_011 badminton_012 badminton_013 badminton_014 badminton_015 badminton_016 badminton_017 badminton_018 badminton_019 badminton_020 badminton_021 badminton_022 badminton_023 badminton_024 badminton_025 badminton_026 badminton_027 badminton_028 badminton_029 badminton_030 badminton_031 badminton_032 badminton_033 badminton_034 badminton_035 badminton_036 badminton_037 badminton_038 badminton_039 badminton_040 badminton_041 badminton_042 badminton_043 badminton_044 badminton_045 badminton_046 badminton_047 badminton_048 badminton_049 badminton_050 badminton_052 badminton_053 badminton_054 badminton_055 badminton_056 badminton_057 badminton_058 badminton_059 badminton_060 badminton_061"
    ["tennis"]="tennis_001 tennis_002 tennis_003 tennis_004 tennis_005 tennis_006 tennis_007 tennis_008 tennis_009 tennis_010 tennis_011 tennis_012 tennis_013"
)

# Human3.6M
sbatch --job-name="h36m_sampling_disabled_nlf" --output="slurm-h36m_sampling_disabled_nlf.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ configs/experiments/ablation_study/sampling/h36m_sampling_disabled_nlf.yaml
sbatch --job-name="h36m_sampling_disabled_rtmpose" --output="slurm-h36m_sampling_disabled_rtmpose.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ configs/experiments/ablation_study/sampling/h36m_sampling_disabled_rtmpose.yaml

# EgoHumans
for seq in "${!EGOHUMANS_SEQUENCES[@]}"; do
    subsequences="${EGOHUMANS_SEQUENCES[${seq}]}"
    sbatch --job-name="egohumans_sampling_disabled_nlf_${seq}" --output="slurm-egohumans_sampling_disabled_nlf_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/sampling/egohumans_sampling_disabled_nlf.yaml "${subsequences}"
    sbatch --job-name="egohumans_sampling_disabled_dwpose_${seq}" --output="slurm-egohumans_sampling_disabled_dwpose_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/sampling/egohumans_sampling_disabled_dwpose.yaml "${subsequences}"
done