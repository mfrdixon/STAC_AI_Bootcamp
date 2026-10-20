# Ordinary Qwen control: patient waiting times

A single frozen sample of 50 synthetic waiting times was reused exactly under ten orders, with three identical repeats. Values were generated once with seed 20261022; orders and randomized execution schedule were frozen in manifest.json before inference. Correct answer: 19 above 30 minutes, 38%, NO (not more than half).

Unmodified Qwen2.5-0.5B-Instruct ran on CPU float32 with eager attention, greedy decoding, repetition penalty 1.0, no tools and no training. A fresh generation configuration was used. All 30 outputs were valid and untruncated.

**All 30 generated outputs were identical:** `NO; count=50; percentage=0%`.

The majority answer was correct but the count and percentage were incorrect and internally inconsistent. Exact correct responses: 0/30. This test does not demonstrate order-induced answer changes.

Teacher-forced YES/NO prefix scores, normalized over those two candidates, gave P(YES) from 0.2374619479 to 0.2679959618: a 3.0534 percentage-point range. EOS was excluded because further count/percentage text follows. These are conditional language probabilities, not calibrated beliefs. The fixed prompt and answer-label order were unchanged; only numerical evidence order varied. Identical-prompt repeat variation was exactly zero for generated text and candidate scores.

Runtime: 48.999 seconds. One sample and one small model do not establish general robustness. No Block Attention model was trained for this task, and no repair claim is supported. Retain this null result for answer-order sensitivity rather than selecting only favorable examples.

## Run
Set QWEN_MODEL_PATH to the local pinned Qwen directory, then run `python run.py`. The code requires torch and transformers and uses offline weights. Revision is recorded in manifest.json. Inputs and raw outputs are included.
