#!/bin/bash

################################################################################
# Multi-Frequency Trading System with Market Regime Classification & CBR
#
# Complete Flow:
# 1. Input data → Market Regime Classification
# 2. Initialize System (Allocator + HFT/MFT/LFT)
# 3. Retrieve optimal strategies from memory (CBR Warm Start)
# 4. RL Training (improve on warm start)
# 5. Update Strategy Memory Bank
################################################################################

echo "================================================================================"
echo "🎯 Multi-Frequency Trading System with Market Regime Classification & CBR"
echo "================================================================================"
echo ""
echo "Architecture:"
echo "  ┌─ 🔍 Market Regime Detector (Top Layer)"
echo "  │  └─ Classifies: HIGH_RISK | HIGH_RETURN | STABLE"
echo "  │"
echo "  ├─ 🧠 Strategy Memory Bank (CBR for all agents)"
echo "  │  ├─ HFT Memory"
echo "  │  ├─ MFT Memory"
echo "  │  ├─ LFT Memory"
echo "  │  └─ Allocator Memory"
echo "  │"
echo "  ├─ 🤖 Allocator Agent (Meta-level PPO)"
echo "  ├─ 📊 HFT Agent (SAC)"
echo "  ├─ 📈 MFT Agent (SAC)"
echo "  ├─ 📉 LFT Agent (SAC)"
echo "  ├─ 🔗 Regime-Adaptive Encoder (LSTM/Transformer/Ensemble)"
echo "  │  ├─ HIGH_RISK → LSTM (fast response)"
echo "  │  ├─ STABLE → Transformer (deep analysis)"
echo "  │  └─ HIGH_RETURN → Ensemble (best of both)"
echo "  ├─ 🛡️  Risk Controller (CVaR, Max Drawdown)"
echo "  └─ 💰 Hedge Manager (Excess → Absolute Return)"
echo ""
echo "================================================================================"
echo ""

# Goal configuration helpers
GOAL_DIR="configs/goals"
DEFAULT_GOAL="$GOAL_DIR/balanced_growth.yaml"

prompt_goal_config() {
    local goal_path=""
    read -p "Use personalized goal config? [y/N]: " use_goal

    if [[ $use_goal == [yY] ]]; then
        if [ -d "$GOAL_DIR" ]; then
            echo ""
            echo "Available goal templates:"
            for file in "$GOAL_DIR"/*; do
                if [ -f "$file" ]; then
                    echo "  - $file"
                fi
            done
            echo ""
            read -p "Enter goal config path [default: $DEFAULT_GOAL]: " goal_path
            goal_path=${goal_path:-$DEFAULT_GOAL}
        else
            echo ""
            read -p "Enter goal config path: " goal_path
        fi

        if [ -n "$goal_path" ] && [ -f "$goal_path" ]; then
            echo "$goal_path"
        else
            echo ""
        fi
    else
        echo ""
    fi
}

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python -c "import numpy, pandas, jax" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Missing dependencies. Installing..."
    pip install numpy pandas jax yfinance matplotlib
    echo ""
fi

# Main Menu
echo "What would you like to do?"
echo ""
echo "Training & Analysis:"
echo "  1) 🔍 Quick Analysis (detect regimes, no training)"
echo "  2) 🚀 Train on ALL regimes (High Risk + High Return + Stable)"
echo "  3) 🎯 Train on SPECIFIC regime"
echo "  4) 📊 Custom data training"
echo "  5) 🧪 Run quick test (100 episodes)"
echo ""
echo "Evaluation:"
echo "  6) 📈 Test evaluation system"
echo "  7) 💡 Run strategy evaluation example"
echo "  8) 🔄 Run integrated workflow (Experiments + Evaluation)"
echo ""
echo "Multi-Frequency & EM Learning:"
echo " 15) 🎯 Multi-Frequency Action Composition Demo"
echo " 16) 🔄 EM Algorithm: Learn Latent Variables for Returns"
echo ""
echo "Risk Control:"
echo " 17) 🛡️  CVaR + Max Drawdown Risk Control Demo"
echo ""
echo "Statistical Validation (Multiple Runs):"
echo " 11) 🧪 Run 30 experiments (Minimum viable - ~30-60 min)"
echo " 12) ✅ Run 50 experiments (Recommended - ~1-2 hours)"
echo " 13) ⭐ Run 100 experiments (High standard - ~3-4 hours)"
echo " 14) 📊 Analyze latest batch results"
echo ""
echo "Status & Info:"
echo "  9) 📚 View memory bank status"
echo " 10) 📊 View evaluation reports"
echo ""
read -p "Enter choice [1-17]: " choice

case $choice in
    1)
        echo ""
        echo "================================================================================"
        echo "🔍 QUICK ANALYSIS MODE"
        echo "================================================================================"
        echo ""
        echo "Analyzing three market regimes:"
        echo "  🔴 High Risk: COVID-19 Crash (2020-02 to 2020-04)"
        echo "  🟢 High Return: Post-COVID Rally (2020-05 to 2021-12)"
        echo "  🟡 Stable: Pre-COVID Market (2019)"
        echo ""

        python -m experiments.run_experiments --experiment predefined
        ;;

    2)
        echo ""
        echo "================================================================================"
        echo "🚀 TRAINING ON ALL REGIMES"
        echo "================================================================================"
        echo ""
        read -p "Number of episodes per regime [default: 500]: " episodes
        episodes=${episodes:-500}

        echo ""
        echo "Training will run in sequence:"
        echo "  1. High Risk regime ($episodes episodes)"
        echo "  2. High Return regime ($episodes episodes)"
        echo "  3. Stable regime ($episodes episodes)"
        echo ""
        echo "⏱️  Estimated time: ~30-60 minutes (depending on hardware)"
        echo ""
        read -p "Continue? [y/N]: " confirm

        if [[ $confirm == [yY] ]]; then
            goal_config=$(prompt_goal_config)
            goal_args=()
            if [ -n "$goal_config" ]; then
                echo "Using goal config: $goal_config"
                goal_args+=(--goal-config "$goal_config")
            fi
            python train.py --mode all_regimes --episodes $episodes "${goal_args[@]}"
        else
            echo "Cancelled."
        fi
        ;;

    3)
        echo ""
        echo "================================================================================"
        echo "🎯 TRAIN ON SPECIFIC REGIME"
        echo "================================================================================"
        echo ""
        echo "Which regime?"
        echo "  1) 🔴 High Risk (COVID-19 Crash)"
        echo "  2) 🟢 High Return (Post-COVID Rally)"
        echo "  3) 🟡 Stable (Pre-COVID 2019)"
        echo ""
        read -p "Enter choice [1-3]: " regime_choice

        case $regime_choice in
            1) regime="high_risk" ;;
            2) regime="high_return" ;;
            3) regime="stable" ;;
            *) echo "Invalid choice"; exit 1 ;;
        esac

        read -p "Number of episodes [default: 500]: " episodes
        episodes=${episodes:-500}

        echo ""
        echo "Training $regime regime with $episodes episodes..."
        goal_config=$(prompt_goal_config)
        goal_args=()
        if [ -n "$goal_config" ]; then
            echo "Using goal config: $goal_config"
            goal_args+=(--goal-config "$goal_config")
        fi
        python train.py --mode auto --regime $regime --episodes $episodes --data-source predefined "${goal_args[@]}"
        ;;

    4)
        echo ""
        echo "================================================================================"
        echo "📊 CUSTOM DATA TRAINING"
        echo "================================================================================"
        echo ""
        read -p "Enter symbols (space-separated, e.g., AAPL MSFT GOOGL): " symbols
        read -p "Start date (YYYY-MM-DD): " start_date
        read -p "End date (YYYY-MM-DD): " end_date
        read -p "Number of episodes [default: 500]: " episodes
        episodes=${episodes:-500}

        echo ""
        echo "Training on custom data..."
        echo "  Symbols: $symbols"
        echo "  Period: $start_date to $end_date"
        echo "  Episodes: $episodes"
        echo ""

        goal_config=$(prompt_goal_config)
        goal_args=()
        if [ -n "$goal_config" ]; then
            echo "Using goal config: $goal_config"
            goal_args+=(--goal-config "$goal_config")
        fi

        python train.py --mode auto \
            --data-source yahoo \
            --symbols $symbols \
            --start $start_date \
            --end $end_date \
            --episodes $episodes \
            "${goal_args[@]}"
        ;;

    5)
        echo ""
        echo "================================================================================"
        echo "🧪 QUICK TEST MODE"
        echo "================================================================================"
        echo ""
        echo "Running quick test with 100 episodes on all regimes..."
        echo "⏱️  Estimated time: ~5-10 minutes"
        echo ""

        goal_config=$(prompt_goal_config)
        goal_args=()
        if [ -n "$goal_config" ]; then
            echo "Using goal config: $goal_config"
            goal_args+=(--goal-config "$goal_config")
        fi

        python train.py --mode all_regimes --episodes 100 --steps 50 "${goal_args[@]}"
        ;;

    6)
        echo ""
        echo "================================================================================"
        echo "📈 TEST EVALUATION SYSTEM"
        echo "================================================================================"
        echo ""
        echo "Testing all evaluation features:"
        echo "  ✅ Single strategy evaluation"
        echo "  ✅ Multi-agent comparison"
        echo "  ✅ Multi-regime comparison"
        echo "  ✅ Comprehensive reports"
        echo ""

        echo "⚠️  Evaluation test scripts have been removed."
        echo "   See docs/evaluation.md for evaluation usage."
        ;;

    7)
        echo ""
        echo "================================================================================"
        echo "💡 STRATEGY EVALUATION"
        echo "================================================================================"
        echo ""
        echo "⚠️  Example scripts have been removed."
        echo "   See docs/evaluation.md for evaluation usage."
        echo "   Use: python evaluation/strategy_evaluator.py"
        ;;

    8)
        echo ""
        echo "================================================================================"
        echo "🔄 INTEGRATED WORKFLOW"
        echo "================================================================================"
        echo ""
        echo "⚠️  Example scripts have been removed."
        echo "   See docs/workflow.md for complete workflow."
        echo "   Use: python train.py --mode auto"
        ;;

    9)
        echo ""
        echo "================================================================================"
        echo "📚 MEMORY BANK STATUS"
        echo "================================================================================"
        echo ""

        if [ -d "memory_bank" ]; then
            echo "Memory bank location: memory_bank/"
            echo ""

            for regime in high_risk high_return stable; do
                echo "📂 $regime:"
                for agent in hft mft lft allocator; do
                    count=$(find memory_bank/$regime/$agent -name "*.pkl" 2>/dev/null | wc -l)
                    if [ $count -gt 0 ]; then
                        echo "  $agent: $count strategies"
                    fi
                done
                echo ""
            done

            total=$(find memory_bank -name "*.pkl" 2>/dev/null | wc -l)
            echo "Total strategies in memory: $total"
            echo ""
            echo "💡 These strategies will be used for warm start in future training!"
        else
            echo "📭 Memory bank is empty (no previous training)"
            echo ""
            echo "Run training to build up the memory bank:"
            echo "  bash run.sh → Choose option 2 or 3"
        fi
        ;;

    10)
        echo ""
        echo "================================================================================"
        echo "📊 VIEW EVALUATION REPORTS"
        echo "================================================================================"
        echo ""

        if [ -d "outputs" ]; then
            echo "Available evaluation reports:"
            echo ""

            # List evaluation directories
            for dir in outputs/*/evaluation; do
                if [ -d "$dir" ] 2>/dev/null; then
                    dirname=$(basename $(dirname $dir))
                    count=$(ls -1 $dir/*.png 2>/dev/null | wc -l)
                    json_count=$(ls -1 $dir/*.json 2>/dev/null | wc -l)

                    if [ $count -gt 0 ] || [ $json_count -gt 0 ]; then
                        echo "📂 $dirname:"
                        echo "   Visualizations: $count PNG files"
                        echo "   Metrics: $json_count JSON files"
                        echo "   Location: $dir/"
                        echo ""
                    fi
                fi
            done

            # Check for monitoring reports
            echo "Available monitoring reports:"
            echo ""
            for dir in outputs/*/monitoring; do
                if [ -d "$dir" ] 2>/dev/null; then
                    dirname=$(basename $(dirname $dir))
                    count=$(ls -1 $dir/*.png 2>/dev/null | wc -l)
                    json_count=$(ls -1 $dir/*.json 2>/dev/null | wc -l)

                    if [ $count -gt 0 ] || [ $json_count -gt 0 ]; then
                        echo "📂 $dirname:"
                        echo "   Monitoring plots: $count PNG files"
                        echo "   Metrics: $json_count JSON files"
                        echo "   Location: $dir/"
                        echo ""
                    fi
                fi
            done

            echo "💡 To view a specific report:"
            echo "   open outputs/<folder>/evaluation/*.png"
            echo "   cat outputs/<folder>/evaluation/*.json | jq"
        else
            echo "📭 No evaluation reports found"
            echo ""
            echo "Run training to generate reports:"
            echo "  bash run.sh → option 2 (Train)"
        fi
        ;;

    11)
        echo ""
        echo "================================================================================"
        echo "🧪 RUN 30 EXPERIMENTS (Minimum Viable Sample Size)"
        echo "================================================================================"
        echo ""
        echo "📊 Statistical Validation with Multiple Independent Runs"
        echo ""
        echo "What you'll get:"
        echo "  • 30 independent training runs (different random seeds)"
        echo "  • Descriptive statistics (mean, std, CI)"
        echo "  • Statistical significance tests (p-value, t-test)"
        echo "  • Effect size (Cohen's d)"
        echo "  • Statistical power analysis"
        echo "  • Bootstrap confidence intervals"
        echo ""
        echo "⏱️  Estimated time: 30-60 minutes"
        echo "💾 Disk space needed: ~500MB"
        echo ""
        read -p "Which regime? [1=high_risk, 2=high_return, 3=stable, 4=all]: " regime_choice

        case $regime_choice in
            1) mode="single"; regime_flag="--regime high_risk" ;;
            2) mode="single"; regime_flag="--regime high_return" ;;
            3) mode="single"; regime_flag="--regime stable" ;;
            4) mode="all_regimes"; regime_flag="" ;;
            *) echo "Invalid choice"; exit 1 ;;
        esac

        read -p "Episodes per run [default: 500]: " episodes
        episodes=${episodes:-500}

        echo ""
        echo "🚀 Starting batch experiments..."
        echo "   Runs: 30"
        echo "   Mode: $mode"
        echo "   Episodes per run: $episodes"
        echo ""

        python experiments/run_multi_experiments.py \
            --n-runs 30 \
            --mode $mode \
            $regime_flag \
            --episodes $episodes

        echo ""
        echo "✅ Batch experiments completed!"
        echo ""
        read -p "Analyze results now? [Y/n]: " analyze

        if [[ ! $analyze == [nN] ]]; then
            python experiments/analyze_multi_experiments.py
        else
            echo ""
            echo "💡 To analyze later, run:"
            echo "   bash run.sh → option 14"
        fi
        ;;

    12)
        echo ""
        echo "================================================================================"
        echo "✅ RUN 50 EXPERIMENTS (Recommended Standard)"
        echo "================================================================================"
        echo ""
        echo "📊 Robust Statistical Validation"
        echo ""
        echo "This is the RECOMMENDED approach for:"
        echo "  • Research papers"
        echo "  • Production deployment decisions"
        echo "  • Reliable performance assessment"
        echo ""
        echo "Benefits:"
        echo "  • ~80% statistical power to detect medium effects"
        echo "  • Reliable confidence intervals"
        echo "  • Robust to outliers"
        echo ""
        echo "⏱️  Estimated time: 1-2 hours"
        echo "💾 Disk space needed: ~1GB"
        echo ""
        read -p "Which regime? [1=high_risk, 2=high_return, 3=stable, 4=all]: " regime_choice

        case $regime_choice in
            1) mode="single"; regime_flag="--regime high_risk" ;;
            2) mode="single"; regime_flag="--regime high_return" ;;
            3) mode="single"; regime_flag="--regime stable" ;;
            4) mode="all_regimes"; regime_flag="" ;;
            *) echo "Invalid choice"; exit 1 ;;
        esac

        read -p "Episodes per run [default: 500]: " episodes
        episodes=${episodes:-500}

        echo ""
        echo "🚀 Starting batch experiments..."
        echo "   Runs: 50"
        echo "   Mode: $mode"
        echo "   Episodes per run: $episodes"
        echo ""

        python experiments/run_multi_experiments.py \
            --n-runs 50 \
            --mode $mode \
            $regime_flag \
            --episodes $episodes

        echo ""
        echo "✅ Batch experiments completed!"
        echo ""
        read -p "Analyze results now? [Y/n]: " analyze

        if [[ ! $analyze == [nN] ]]; then
            python experiments/analyze_multi_experiments.py
        else
            echo ""
            echo "💡 To analyze later, run:"
            echo "   bash run.sh → option 14"
        fi
        ;;

    13)
        echo ""
        echo "================================================================================"
        echo "⭐ RUN 100 EXPERIMENTS (High Standard)"
        echo "================================================================================"
        echo ""
        echo "📊 Publication-Quality Statistical Validation"
        echo ""
        echo "This is the GOLD STANDARD for:"
        echo "  • Academic publications"
        echo "  • High-stakes commercial decisions"
        echo "  • Detecting small but important effects"
        echo ""
        echo "Benefits:"
        echo "  • ~90% statistical power"
        echo "  • Very narrow confidence intervals"
        echo "  • Can detect small improvements"
        echo "  • Maximum confidence in results"
        echo ""
        echo "⚠️  WARNING: This will take significant time and resources!"
        echo "⏱️  Estimated time: 3-4 hours"
        echo "💾 Disk space needed: ~2GB"
        echo ""
        read -p "Which regime? [1=high_risk, 2=high_return, 3=stable, 4=all]: " regime_choice

        case $regime_choice in
            1) mode="single"; regime_flag="--regime high_risk" ;;
            2) mode="single"; regime_flag="--regime high_return" ;;
            3) mode="single"; regime_flag="--regime stable" ;;
            4) mode="all_regimes"; regime_flag="" ;;
            *) echo "Invalid choice"; exit 1 ;;
        esac

        read -p "Episodes per run [default: 500]: " episodes
        episodes=${episodes:-500}

        echo ""
        echo "⚠️  This will run 100 independent training sessions!"
        echo ""
        read -p "Are you sure? [y/N]: " confirm

        if [[ ! $confirm == [yY] ]]; then
            echo "Cancelled."
            exit 0
        fi

        echo ""
        echo "🚀 Starting batch experiments..."
        echo "   Runs: 100"
        echo "   Mode: $mode"
        echo "   Episodes per run: $episodes"
        echo ""

        python experiments/run_multi_experiments.py \
            --n-runs 100 \
            --mode $mode \
            $regime_flag \
            --episodes $episodes

        echo ""
        echo "✅ Batch experiments completed!"
        echo ""
        read -p "Analyze results now? [Y/n]: " analyze

        if [[ ! $analyze == [nN] ]]; then
            python experiments/analyze_multi_experiments.py
        else
            echo ""
            echo "💡 To analyze later, run:"
            echo "   bash run.sh → option 14"
        fi
        ;;

    14)
        echo ""
        echo "================================================================================"
        echo "📊 ANALYZE BATCH EXPERIMENT RESULTS"
        echo "================================================================================"
        echo ""

        # Find available batches
        if [ -d "outputs/multi_run_experiments" ]; then
            echo "Available batch results:"
            echo ""

            batch_count=0
            for batch_dir in outputs/multi_run_experiments/batch_*; do
                if [ -d "$batch_dir" ]; then
                    batch_count=$((batch_count + 1))
                    batch_name=$(basename $batch_dir)

                    # Extract info from summary if exists
                    if [ -f "$batch_dir/batch_summary.json" ]; then
                        n_runs=$(python -c "import json; print(json.load(open('$batch_dir/batch_summary.json'))['summary']['total_runs'])" 2>/dev/null || echo "?")
                        success=$(python -c "import json; print(json.load(open('$batch_dir/batch_summary.json'))['summary']['successful_runs'])" 2>/dev/null || echo "?")

                        echo "  $batch_count) $batch_name"
                        echo "      Total runs: $n_runs | Successful: $success"
                    else
                        echo "  $batch_count) $batch_name"
                    fi
                    echo ""
                fi
            done

            if [ $batch_count -eq 0 ]; then
                echo "📭 No batch results found."
                echo ""
                echo "Run batch experiments first:"
                echo "  bash run.sh → option 11, 12, or 13"
                exit 0
            fi

            echo ""
            read -p "Analyze which batch? [1-$batch_count, or 0 for latest]: " batch_choice

            if [ "$batch_choice" = "0" ] || [ -z "$batch_choice" ]; then
                echo ""
                echo "Analyzing latest batch..."
                python experiments/analyze_multi_experiments.py
            else
                # Get the Nth batch directory
                batch_dir=$(ls -1d outputs/multi_run_experiments/batch_* 2>/dev/null | sed -n "${batch_choice}p")

                if [ -z "$batch_dir" ]; then
                    echo "Invalid choice"
                    exit 1
                fi

                echo ""
                echo "Analyzing: $(basename $batch_dir)"
                python experiments/analyze_multi_experiments.py --results-dir "$batch_dir"
            fi

            echo ""
            echo "✅ Analysis completed!"
            echo ""
            echo "📊 Generated files:"
            echo "   • statistical_analysis_results.json"
            echo "   • comprehensive_analysis.png"
            echo "   • rl_vs_baseline_comparison.png"
            echo "   • statistical_validation/ (detailed reports)"

        else
            echo "📭 No batch experiments found."
            echo ""
            echo "Run batch experiments first:"
            echo "  bash run.sh → option 11, 12, or 13"
        fi
        ;;

    15)
        echo ""
        echo "================================================================================"
        echo "🎯 MULTI-FREQUENCY ACTION COMPOSITION DEMO"
        echo "================================================================================"
        echo ""
        echo "This demo shows:"
        echo "  1. What actions each frequency agent outputs"
        echo "  2. How actions are converted to portfolio weights"
        echo "  3. Multi-frequency coordination formula"
        echo "  4. Complete timestep execution flow"
        echo "  5. Portfolio performance evaluation"
        echo ""
        echo "Agents:"
        echo "  • HFT: Order-level actions (6D)"
        echo "  • MFT: Position-level actions (2D)"
        echo "  • LFT: Portfolio-level actions (num_assets D)"
        echo "  • Allocator: Capital allocation [α_H, α_M, α_L]"
        echo ""
        echo "Formula: π* = α_H·π_H* + α_M·π_M* + α_L·π_L*"
        echo ""

        echo "⚠️  Example scripts have been removed."
        echo "   See docs/multi_frequency_actions.md for details."
        echo "   Multi-frequency action composition is integrated in train.py"
        ;;

    16)
        echo ""
        echo "================================================================================"
        echo "🔄 EM ALGORITHM: LEARN LATENT VARIABLES FOR RETURNS"
        echo "================================================================================"
        echo ""
        echo "⚠️  Example scripts have been removed."
        echo "   See docs/em_training.md for EM algorithm usage."
        echo "   EM training is integrated in shared_encoder/em_training.py"
        ;;

    17)
        echo ""
        echo "================================================================================"
        echo "🛡️  CVaR + MAX DRAWDOWN RISK CONTROL"
        echo "================================================================================"
        echo ""
        echo "⚠️  Example scripts have been removed."
        echo "   See risk_controller/cvar_drawdown_controller.py for usage."
        echo "   Risk control is integrated in train.py"
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "✅ DONE"
echo "================================================================================"
echo ""
echo "📁 Output files:"
echo "  outputs/          - Training results, evaluation reports, and logs"
echo "  memory_bank/      - Strategy cases for future warm starts"
echo ""
echo "Next steps:"
echo "  • View training results: cat outputs/*_results.json"
echo "  • View evaluation reports: bash run.sh → option 10"
echo "  • Check memory bank: bash run.sh → option 9"
echo "  • Test evaluation: bash run.sh → option 6"
echo "  • See docs/ for usage examples"
echo "  • Multi-frequency demo: bash run.sh → option 15"
echo "  • EM learning: bash run.sh → option 16"
echo "  • Risk control demo: bash run.sh → option 17"
echo "  • Statistical validation: bash run.sh → option 11-14"
echo ""
echo "💡 Tips:"
echo "   • Each training run improves the memory bank!"
echo "   • Evaluation reports include 30+ performance metrics"
echo "   • See docs/ for detailed documentation"
echo "   • Multi-frequency demo shows how agents compose actions"
echo "   • EM algorithm learns latent variables that explain returns"
echo "   • Risk control focuses on CVaR + Max Drawdown (practical, not statistical)"
echo "   • For robust results, run 30-50 experiments (options 11-12)"
echo "   • Statistical validation proves RL superiority scientifically"
echo ""
echo "================================================================================"
