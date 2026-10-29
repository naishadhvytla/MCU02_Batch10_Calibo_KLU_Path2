# Credit Pre-Screening Engine — Loan Intake Optimisation

**Calibo AI Academy · Path 2 · KLU Batch 10 · Phase 2 — Mini Use Case 02 (MUC02) · Supervised Learning Part 1**

Six months of personal-loan decisions turned into an auto-decline threshold a bank can defend, plus **Triage**, a live web app that runs the trained model in the browser.

**Live app:** https://triagecreditscreener.netlify.app · **App source:** [Triage---Credit-PreScreening-Engine](https://github.com/naishadhvytla/Triage---Credit-PreScreening-Engine)

---

## The question

A mid-size private bank receives about **3,000 personal loan applications a month**, and an analyst reviews every one for about **25 minutes**. Nearly half are declined, many for reasons visible in the first minute. The Head of Retail Lending wants to know whether a model can take the clear-cut declines out of the manual queue.

> **Can a pre-screening model safely remove clear-cut declines from manual review, and how many applications and analyst hours would that free each month, without wrongly declining more than 5% of the applicants it auto-declines?**

## Headline findings

| Metric | Value | What it means |
| --- | --- | --- |
| Applications analysed | 12,000 · 52.9% approved | Apr–Sep 2026; a balanced target |
| Credit score floor | **3.0% approved below 650** | Jumps to 32.5% at 650–699: an unwritten rule |
| Affordability cliff | **65.5% → 15.6%** | Approval from FOIR under 30% to FOIR above 55% |
| Model at the default 0.5 | 79.1% accurate · AUC 0.867 | But **225** good customers in the test set would be declined |
| Recommended lane | **0.90 + new-to-credit to analysts** | ~441 applications and ~184 analyst hours a month, 3.1% wrong |

**The conclusions:**

1. **The bank already runs on two unwritten rules.** Below a score of 650 analysts almost never approve, and once EMIs take more than half of income (FOIR above 50%) approvals collapse. The model's strongest coefficients, FOIR (+1.22), credit score (−1.01) and late payments (+0.98), show it learned this policy rather than noise.
2. **Accuracy is the wrong headline.** At the default cut-off, one auto-decline in five would be a good customer. The bank's 5% rule, not the model, picks the threshold: **0.90** is the lowest that meets it (15.2% auto-declined, 4.7% wrong).
3. **New-to-credit applicants must stay with humans.** The model is 56.7% accurate for them (81.2% for scored applicants), and half of those it would auto-decline at 0.90 were approved. Routing all of them to analysts cuts wrong declines to **3.1%** for half a point of volume.
4. **The model does not widen the gender gap.** The historical 3.7-point gap (54.1% vs 50.4%) comes mostly from income and job mix; auto-decline rates are 15.1% for women and 14.5% for men. Income and job type can still act as proxies, so outcomes should be monitored monthly.

With new-to-credit applicants excluded, 0.85 also passes the rule (4.8%) and would free about 238 hours a month, but a 0.2-point margin on 2,400 test applications is too thin for a first launch.

---

## Submission documents

All write-ups sit at the top level of the repo:

| Document | What it is |
| --- | --- |
| [Batch10_MUC02_KLU_Summary.docx](Batch10_MUC02_KLU_Summary.docx) | Team documentation & insight report |
| [Batch10_MUC02_KLU_Summary_1Page.docx](Batch10_MUC02_KLU_Summary_1Page.docx) | 1-page submission summary (≤500 words) |
| [Batch10_MUC02_KLU_Individual_Report_Naishadh_Vytla.docx](Batch10_MUC02_KLU_Individual_Report_Naishadh_Vytla.docx) | Individual report: Naishadh Vytla, Data Analyst |
| [Batch10_MUC02_KLU_Individual_Report_Raghunandan.docx](Batch10_MUC02_KLU_Individual_Report_Raghunandan.docx) | Individual report: Raghunandan, Insight Analyst |
| [Batch10_MUC02_KLU_Individual_Report_Pradeepthi.docx](Batch10_MUC02_KLU_Individual_Report_Pradeepthi.docx) | Individual report: Pradeepthi, Model Builder |
| [Batch10_MUC02_KLU_Individual_Report_Saketh.docx](Batch10_MUC02_KLU_Individual_Report_Saketh.docx) | Individual report: Saketh, Policy Synthesiser |

## Repository structure

```
├── Batch10_MUC02_KLU_Summary.docx              # team documentation & insight report
├── Batch10_MUC02_KLU_Summary_1Page.docx        # ≤500-word submission summary
├── Batch10_MUC02_KLU_Individual_Report_*.docx  # one per member (4)
├── notebook/
│   └── Batch10_MUC02_KLU_Notebook.ipynb        # CBIM canvas + Steps 5.1–5.7 (runs top-to-bottom, 0 errors)
├── data/
│   └── MUC02_Credit_PreScreening_Dataset.csv   # 12,000 applications · 19 columns · Apr–Sep 2026
├── deck/
│   └── Batch10_MUC02_KLU_Presentation.pptx     # 13 slides, Triage walkthrough video on slide 12
├── model/
│   ├── train_export.py                         # retrains the notebook pipeline → app/model.json
│   └── parity_fixture.json                     # test rows + scikit-learn probabilities for the JS parity check
└── app/                                        # Triage Lite: offline single-file version (no accounts)
    ├── index.html                              # the whole app, one file (model embedded)
    ├── model.json                              # exported coefficients, scaler, encoder, test predictions
    ├── sample_batch.csv                        # 240-application sample for the Batch view
    └── src/                                    # body.html + build.py that assemble index.html
```

The full deployed Triage (accounts, Supabase, tests) lives in its own repo: [Triage---Credit-PreScreening-Engine](https://github.com/naishadhvytla/Triage---Credit-PreScreening-Engine).

---

## Running the notebook

**Google Colab:**
1. Open [colab.research.google.com](https://colab.research.google.com) → **Upload notebook** → `notebook/Batch10_MUC02_KLU_Notebook.ipynb`
2. Upload `data/MUC02_Credit_PreScreening_Dataset.csv` to the session root
3. **Runtime → Run all**

**Locally:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook notebook/Batch10_MUC02_KLU_Notebook.ipynb
```

The notebook finds the CSV beside itself or in `../data/`, so both setups work unchanged.

### Analysis steps

| Step | Section | Owner |
| --- | --- | --- |
| — | CBIM Problem Canvas | All four members (led by Naishadh Vytla) |
| 5.1 | Load, inspect & clean | Naishadh Vytla |
| 5.2 | What drives approval today? | Raghunandan |
| 5.3 | Affordability features (FOIR, loan-to-income, new-to-credit flag) | Raghunandan |
| 5.4 | Logistic regression pre-screener | Pradeepthi |
| 5.5 | Auto-decline threshold & business case | Pradeepthi |
| 5.6 | Fairness check | Saketh |
| 5.7 | Credit Policy Briefing | Saketh |

### Method note — the SSOT

Every chart, the model and the app read from **one** cleaned frame built in Step 5.1. Each cleaning decision leaves a flag (`income_fixed_annual`, `income_was_missing`, `is_ntc`) and **no rows are dropped**. The target column records **analyst decisions, not repayment**: the model learns to copy past judgement, which is the limitation everything else is read against.

---

## Triage — Web App

Built by **Naishadh Vytla** for MUC02, a sibling of [Clarity](https://clarityeda.netlify.app) from MUC01.

| | |
| --- | --- |
| **Live** | https://triagecreditscreener.netlify.app |
| **Source** | [Triage---Credit-PreScreening-Engine](https://github.com/naishadhvytla/Triage---Credit-PreScreening-Engine) |
| **Model** | `muc02-logreg-1`, trained in this repo's notebook pipeline and shipped as `model.json` |

Triage takes clear-cut declines out of the manual queue. **The model never approves.** Every application lands in one of three lanes:

| Lane | When |
| --- | --- |
| **Auto-decline** | decline confidence ≥ the policy threshold (default 0.90) **and** the applicant has a bureau score |
| **Analyst review** | everything else |
| **Analyst only** | applicants with no credit history (routed to analysts by default) |

**Landing → Sign in → Score → Threshold → Batch → Report → History**

* **Landing** with a live demo that scores one example application in place
* **Sign in** with email + password or Google; create account, forgot / reset password
* **Score** one application → lane, confidence, FOIR and loan-to-income, the factors that moved the score; save it to History with a note
* **Threshold** explorer and fairness monitor; *Save as my policy* stores your threshold and new-to-credit routing, which Score and Batch then use
* **Batch**: upload an intake CSV (the bank's 13 columns); it is scored in the browser, stored privately in your Storage folder, and saved as a batch record
* **Report**: a visual summary (KPIs, the 5% rule, lanes, confidence histogram) and a numbered analyst report, exportable as Markdown, HTML, PDF or the scored CSV
* **History**: up to **50 scorings** and **5 batches** per user, enforced in the database; every save, delete and policy change goes to an append-only audit log
* **About the model**: the model card; account menu with sign out and delete account

**Architecture.** Static pages on Netlify (CSP, HSTS) with no server of our own; scoring runs in the browser from one shared `js/model.js`. Supabase provides Auth, Postgres with row-level security (profiles · policies · scorings · batches · audit_log), triggers for limits and auditing, and a private `intake/{user_id}/` Storage bucket. Only the anon key is in the client.

**Tests** (in the Triage repo):

| Test | What it checks |
| --- | --- |
| `node tests/parity.mjs` | JS scorer matches scikit-learn's `predict_proba` within 1e-8 on 300 / 300 fixture rows, plus lane and cleaning rules |
| `bash tests/db/run.sh` | RLS isolation between two users, the 50 / 5 limits, append-only audit log, storage folder rules, account deletion |
| `npx playwright test` | Smoke tests at desktop and 360px: landing, login, redirects, 404, keyboard, theme; signed in: score → save → History, batch upload → lane counts |

Setup (Supabase, Google sign-in, Netlify) is documented in the [Triage README](https://github.com/naishadhvytla/Triage---Credit-PreScreening-Engine#setup).

**Triage Lite** (`app/` in this repo) is the original single-file version: Score, Threshold, Batch and the model card with no accounts, running offline from `app/index.html`. Rebuild it after changing the model:

```bash
python model/train_export.py   # retrain + export app/model.json
python app/src/build.py        # reassemble app/index.html
```

---

## Future work

- **Train on repayment, not opinions.** Joining approved loans to their 12-month repayment record would show whether the 650 floor and 50% FOIR limit are the right rules or just habitual ones.
- **Shadow-mode launch.** Score live applications for a month without declining any; if wrong declines stay well under 5%, test 0.85 (about 54 more analyst hours a month).
- **Monitor fairness after launch.** Report approval and auto-decline rates by gender monthly; the residual gaps inside Salaried (2.3 pts) and Self-Employed (4.3 pts) need watching.

## Team — Batch 10

| Member | Role |
| --- | --- |
| **Naishadh Vytla** | Data foundation & cleaning · notebook assembly · Triage web app ([source](https://github.com/naishadhvytla/Triage---Credit-PreScreening-Engine)) |
| **Raghunandan** | Approval drivers & affordability features · chart consistency · team documentation |
| **Pradeepthi** ([@PradeepthiPathuri](https://github.com/PradeepthiPathuri)) | Model & threshold · presentation deck |
| **Saketh** | Fairness & Credit Policy Briefing · QA · 1-page summary |

Each member's individual report (listed under [Submission documents](#submission-documents)) covers their scope, method, findings, business interpretation, what they learned and one limitation.

---

## Limitations

- **The target is past analyst judgement, not repayment.** The model can copy bias in past decisions and cannot say whether auto-declined applicants would have repaid.
- **Declared income is not verified**, and 441 incomes (mostly self-employed) were filled with job-type medians, so their FOIR is approximate.
- **One 80/20 split, no cross-validation** (not yet in the curriculum); threshold differences of a few tenths of a point are within sampling noise.
- **Gender is recorded only as Male/Female**, and income and job type can act as proxies.
- Six months of one bank's simulated applications; re-validate on newer data before any live use.

---

*Academic coursework for Calibo AI Academy. The dataset is synthetic and provided as part of the MUC02 brief. Triage makes no lending decision about real people.*
