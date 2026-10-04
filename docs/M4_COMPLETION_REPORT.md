# M4 Completion Report

We have successfully completed the full M4 XLM-R training on the Victus RTX 4050.

## Training Metrics & Performance
- **Binary Test Macro-F1**: 0.7610
- **Oracle Category Macro-F1**: 0.5478
- **Cascaded Category Macro-F1**: 0.5072
- **Training Duration**: 24.82 minutes
- **Peak GPU Memory**: ~5392.79 MB
- **Truncation >128 tokens**: 0.84% of examples (54 instances) exceeded the max length of 128.

## Insights
- **Major Category Weakness**: Violence and Others. The model struggled to confidently predict these categories, emphasizing the need for targeted data augmentation or further architectural tuning in future milestones.

## Disclaimers
*M4 is a research baseline/component. The metrics produced reflect a baseline capability, but this is NOT a claim of a production-ready hate detection system.*

## Outputs
- **Experiment Metadata**: Stored automatically under `experiments/`.
- **Aggregate Error Analysis**: Computed and stored in `reports/m4_aggregate_error_analysis.json`.
- **Checkpoints**: All checkpoints (`xlmr_best.pt`, `xlmr_resumable_best.pt`, `xlmr_latest.pt`) are preserved in `models/`.
