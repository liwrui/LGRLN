
#!/usr/bin/bash
# Training
python tools/train_context_reasoning.py --cfg configs/video-summarization/TVSum/SPELL_default.yaml --split 1
python tools/train_context_reasoning.py --cfg configs/video-summarization/TVSum/SPELL_default.yaml --split 2
python tools/train_context_reasoning.py --cfg configs/video-summarization/TVSum/SPELL_default.yaml --split 3
python tools/train_context_reasoning.py --cfg configs/video-summarization/TVSum/SPELL_default.yaml --split 4
python tools/train_context_reasoning.py --cfg configs/video-summarization/TVSum/SPELL_default.yaml --split 5
# Evaluation
python tools/evaluate.py --exp_name SPELL_VS_TVSum_default --eval_type VS_avg --all_splits
