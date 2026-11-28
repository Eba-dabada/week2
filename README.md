# Bank Marketing: Who Should the Bank Call?

**Answer:** call the 1,362 clients the model flags (15.1% of the list). That reaches 594 of the 1,058 subscribers in the held-out test set (56.1% recall, 43.6% precision, test F1 **0.4909**). F1 is the metric because only 11.7% of clients subscribe, so accuracy says nothing: always predicting "no" scores 88.3%.

## Problem

A team has more cases than it can deal with. Most of them do not matter and a few matter a great deal. The team can only act on some of them, or acting wrongly costs one amount and missing a real case costs far more. They have historical data, and they have asked you one question: which cases should we act on, and what do we get for it?

You are to build a classifier on a real, imbalanced table of data and turn its scores into that decision. "The model is 97% accurate" is not an answer here: on a table like this, a model that always says no can score that too. Your answer is a statement the team can act on, with a number behind it that survives a reviewer asking how you got it.

A bank's telemarketing team has far more clients than it can call. Most will not subscribe to a term deposit and a few will. Calling the wrong client costs one amount, and missing a real subscriber costs more. The team has historical campaign data and one question: **which clients should we call, and what do we get for it?**

The data is imbalanced (88.3% no, 11.7% yes), so a model that always says "no" is "88% accurate" and finds nobody. This project builds a classifier on that table and turns its scores into a call list, using a decision threshold chosen on validation data. The model is judged on precision, recall and F1 for the "yes" class.

- **Data:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), `bank-full.csv`, downloaded automatically and checked against a pinned SHA-256 hash. 45,211 rows, 15 features plus the target `y`.
- **Dropped column:** `duration`. Call length is only known after the call ends, so using it would leak the outcome.
- **Split:** 60% train (27,126 rows) / 20% validation (9,042) / 20% test (9,043, with 1,058 positives), stratified, `random_state=42`. The split is identical on every run.
- **Preprocessing:** one-hot encoding, then `StandardScaler` fitted on the training rows only (41 input features). The column list, scaler statistics and threshold are saved in the checkpoint, so evaluation never refits anything.
- **Model:** MLP with one hidden layer (41 units, ReLU, dropout 0.3), `BCEWithLogitsLoss` with `pos_weight` for the imbalance, Adam at lr 0.001.
- **Checkpoint:** the best epoch by validation loss (epoch 65 of 100) is kept, not the last one.
- **Threshold:** chosen on validation F1 over 0.3 to 0.7. It came out at **0.60**.
- **Review limit:** I assumed the team can call about 1,500 clients. This is my choice and is not part of the dataset.

## Results

**Claim:** `test_f1` = **0.4909**, tolerance 0.006 (absolute), measured on the released checkpoint: [`best_model.pt`, release v2.0.0](https://github.com/Eba-dabada/week2/releases/download/v2.0.0/best_model.pt). CI downloads that file on every pull request, evaluates it on a CPU-only runner, and fails if the result differs from the claim by more than the tolerance.

| Model | Split | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Trivial (majority class) | validation | n/a | 0.883 | 0.000 | 0.000 | 0.000 |
| HistGradientBoosting (balanced) | validation | 0.50 | 0.823 | 0.359 | 0.646 | 0.461 |
| **MLP (released checkpoint)** | **test** | **0.60** | **0.864** | **0.436** | **0.561** | **0.491** |

Always predicting "no" scores 0.8830 accuracy on the test set. The MLP scores 0.8638, which is lower, so accuracy is not used to judge the model.

**Test confusion matrix (n = 9,043):**

| | Actual yes | Actual no |
|---|---|---|
| **Predicted yes** | 594 (TP) | 768 (FP) |
| **Predicted no** | 464 (FN) | 7,217 (TN) |

The confusion matrix shows what happened when the model's predictions were compared with the customers' actual answers.

- **True Positive (TP):** 594 customers were correctly identified as subscribers.

- **False Positive (FP):** 768 customers were selected but did not subscribe.

- **False Negative (FN)**: 464 customers subscribed, but the model failed to identify them.

- **True Negative (TN):** 7,217 customers did not subscribe and were correctly skipped.

### What the Team Gets

Call the 1,362 customers selected by the model (15.1% of the entire list).

- **594 actual subscribers:** The model successfully found 594 of the 1,058 customers who subscribed.

- **56.1% Recall:** Out of all customers who subscribed, the model found 56.1%.

- **43.6% Precision:** Out of all customers selected by the model, 43.6% actually subscribed.

- **3.7 times more likely:** The selected customers are about 3.7 times more likely to subscribe than a randomly chosen customer, compared with the original 11.7% subscription rate.

In simple terms, instead of calling everyone, the bank can focus on the 1,362 customers selected by the model. However, some selected customers will not subscribe, and some actual subscribers will still be missed.

**Cost break-even:** let V be the value of one subscriber and C the cost of one call. The model beats calling no one when V/C > about 2.3 (594V > 1,362C), and beats calling everyone when V/C < about 16.6 (464V < 7,681C). Inside that range it is the best of the three policies. This project does not know the real V/C, and these figures do not show that 0.60 is the best threshold for any particular ratio.

### Experiment Log

| # | Run | What changed and why | What happened |
|---|---|---|---|
| 1 | Smoke run (512 rows, 2 epochs) | Added `--smoke` to check the whole loop in seconds before a long run | Ran in seconds. Validation F1 was about 0.23, which means nothing for a 2-epoch model, but the loop, checkpoint and `evaluate` worked end to end |
| 2 | Full run, 100 epochs | `pos_weight` for the imbalance, dropout 0.3, Adam at lr 0.001, checkpoint kept by validation loss | Best epoch 91 (validation loss 0.9732), threshold 0.70, validation F1 0.4784. Released as v1.0.0, test F1 0.4916 |
| 3 | Threshold sweep on validation: 0.3, 0.4, 0.5, 0.6, 0.7 (part of every full run) | 0.5 is not necessarily the best cut for an imbalanced target, so I compared thresholds on validation F1 | Run 2 chose 0.70, run 5 chose 0.60. I never searched beyond 0.70 |
| 4 | Baselines: always "no" and HistGradientBoosting (balanced) | A reference point on the same split | Always "no" scores 0 F1. Gradient boosting's validation F1 is 0.461, just under the MLP's |
| 5 | Retrain, 100 epochs, same code and seed 42 | Repeat of run 2 to see whether it reproduces | Best epoch 65 (validation loss 0.9734), threshold 0.60, validation F1 0.4748. The same seed gave a different result, and I did not isolate why (my library versions changed in between, which is one possible cause) |
| 6 | Released run 5 as v2.0.0 and made it the claim | I chose to replace v1.0.0 after seeing its test score | Test F1 0.4909 against 0.4916 for v1.0.0 |

## How to Use

Requires Python 3.12. Everything runs on CPU.

### 1. Install

Requires **Python 3.12**. Check with `python --version`. If you don't have it, download it from https://www.python.org/downloads/ (on Windows, tick "Add python.exe to PATH" in the installer).

Open a terminal in the project folder, then run the commands for your system.

**Windows (PowerShell or Command Prompt)**

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

**Windows (Git Bash)**

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Then, on every system:

```bash
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Check that it worked:

```bash
python -c "import torch; print(torch.__version__)"
```

You should see a version number with no errors. No GPU is needed. From here on, keep the environment active (you should see `(.venv)` at the start of the terminal line).

### 2. Tests

```bash
python -m pytest
```

### 3. Write the split (train / validation / test row ids)

```bash
python -m src.split --out runs/split.json
```

### 4. Baselines (always "no" and gradient boosting)

```bash
python -m src.baseline
```

### 5. Smoke training run (512 rows, 2 epochs)

```bash
python -m src.train --smoke --out runs/smoke.pt
```

### 6. Full training (best epoch by validation loss, threshold chosen on validation)

```bash
python -m src.train --out runs/best_model.pt --epochs 100 --plot
```

### 7. Evaluate the released checkpoint once on the test set

```bash
mkdir -p runs
curl -L --fail -o runs/released.pt https://github.com/Eba-dabada/week2/releases/download/v2.0.0/best_model.pt
python -m src.evaluate --checkpoint runs/released.pt --metrics-out runs/metrics.json
```

- `runs/metrics.json` contains `test_f1`, which should be 0.4909 within 0.006.
- The commands CI runs are listed under `[tool.reproduce]` in `pyproject.toml`.
- Seeds are fixed, and the test set is only used in step 7.
- The checkpoint is a release asset and is not committed to the repository.
- Retraining from scratch (step 6) can give a different result, so the claim applies to the released file.

## Where It Fails

- **It misses 43.9% of subscribers** (464 of 1,058 in the test set).
- **Its calls are wrong 56.4% of the time:** 768 of the 1,362 clients it flags did not subscribe.
- **Its lead over gradient boosting is small** (validation F1 0.475 against 0.461), and it comes from one split and one seed, so I cannot say it is a real difference.
- **Validation loss stopped improving early.** Train loss kept falling to about 0.923 by epoch 100, while validation loss stayed between about 0.973 and 0.980 from epoch 25 onwards.

### Leakage Audit

**Checked and handled**

- `duration` is only known after a call ends, so it is dropped.
- The split happens before anything is fitted. The scaler is fitted on training rows only and its statistics are saved in the checkpoint.
- The threshold and the best epoch are both chosen on validation. The test set is scored once per checkpoint, on the final model.
- The split row ids are written to JSON. Two runs give identical files, and train and test do not overlap.

**Not checked**

- I did not verify that `campaign`, `pdays`, `previous` and `poutcome` would be known when a call is planned.

## Limitations

- **One split and one seed (42).** I did not train with other seeds, so I cannot say how big a difference has to be before it is real. The MLP's lead over gradient boosting is not established.
- **Only one way of handling the imbalance.** I used `pos_weight` and did not compare it with another approach, so I cannot say how much it helped.
- **The threshold sits inside a small grid.** I searched 0.3 to 0.7. The first run chose 0.70, the top of the grid, so a higher value might have scored better.
- **The threshold maximises F1, not profit.** The dataset gives no cost for a call or value for a subscriber, so the break-even range above does not show that 0.60 is the best cut for any particular ratio.
- **The review limit is my assumption.** I assumed the team can call about 1,500 clients. The model flags 1,362.
- **I replaced the first checkpoint after seeing its test score.** I released v1.0.0 (test F1 0.4916), then retrained and released v2.0.0 (test F1 0.4909) as the claim. Because that decision came after seeing a test number, the reported test score is slightly optimistic. The gap is 0.0007, so the effect is small.
- **Same seed, different result.** Retraining gave a different best epoch (65 against 91) and threshold (0.60 against 0.70), so the released checkpoint is the only thing the claim covers.
- **No per-segment analysis.** I do not know which groups the model serves badly.
- **Resume is not tested.** `train.py` has a `load_checkpoint` function, but I have no test showing that an interrupted run ends where an uninterrupted one would, and none runs in CI.
- **One dataset.** It comes from one Portuguese bank's telemarketing campaigns and may not transfer to other banks, countries or years.
- **The tolerance covers small numeric differences, not retraining.** 0.006 allows for CPU, GPU and library differences when evaluating the released checkpoint.

## AI Use

- **Claude**
  - Tutor: explained concepts and guided me step by step through the project contract.
  - Syntax and logic review: pointed out errors in my code, which I fixed myself.
  - Wrote the boilerplate for the CI workflow.
  - Drafted and formatted this README.
- **Gemini**
  - Syntax review.
  - Explained concepts I was not familiar with.
  - Instructed to give no code.
- **Written and checked by me**
  - The training loop, the checkpoint code and the choice of threshold on validation.
  - The split and the evaluation script.
  - Every command was run by me, and the numbers in Results come from my runs.
  - All decisions were made by me.