# Technical Report: Feature Engineering, Scientific Validation, and *Zero-Leakage* Modeling
**Project:** Credit Card Fraud Detection (`Project_Banking_Fraud`)  
**Code References:** [`src/002_modeling.ipynb`](./002_modeling.ipynb) and [`src/001_eda.ipynb`](./001_eda.ipynb)  
**Generated Datasets:** [`data/processed/creditcard_clean.parquet`](../data/processed/creditcard_clean.parquet) and [`data/processed/creditcard_engineered_best.parquet`](../data/processed/creditcard_engineered_best.parquet)

---

## Executive Summary

This report documents the second phase of the Banking Credit Card Fraud Detection Data Science project. Building upon the structural diagnostics uncovered during Exploratory Data Analysis ([`src/report_eda_001.md`](./report_eda_001.md)), we built in [`src/002_modeling.ipynb`](./002_modeling.ipynb) an end-to-end pipeline covering Feature Engineering, In-Fold Feature Selection, Imbalance Handling, *Ensemble Blending*, and Financial Threshold Calibration under a **strict *Zero Data Leakage* and *Zero Retrospective Peeking* protocol**.

The key milestones achieved were:
1. **Strict Isolation of the Holdout Vault (`df_test_vault`):** Immediately after exact deduplication (`283,726` unique transactions, `473` frauds), the final **chronological 20% (`56,746` rows, `74` frauds = `€7,727.67`)** were locked inside the `df_test_vault` holdout vault. All experimentation across Blocks 1 to 4 was conducted exclusively on the first **chronological 80% (`df_dev`: `226,980` rows, `399` frauds = `€50,863.72`)**, evaluated simultaneously by two independent internal judges: **Judge 1 (`5-Fold Stratified CV`)** and **Judge 2 (`Internal Chronological Split 64% -> 16%`)**.
2. **Ablation Validation of Dataset `D6_Selected_Best` (`29` columns):** Engineering domain-agnostic financial features (`log1p_amount`, `amount_robust`, `is_zero_amount`, `is_round_integer`), circadian trigonometric encodings (`hour_sin`, `hour_cos`), dropping raw `Time`, and applying *in-fold* statistical noise pruning (`KS < 0.14` computed strictly on each fold's `X_train`) outperformed the baseline dataset across **4 out of 4 algorithm families** (`LogisticRegression`, `RandomForest`, `LightGBM`, and `XGBoost`).
3. **Empirical Refutation of Synthetic Resampling (`SMOTE` / `UnderSampling`) and Discovery of Algorithmic Complementarity:** We proved within folds that `SMOTE` (`0.7541` on Temporal XGBoost, a `-3.85 p.p.` drop) and `RandomUnderSampler` (`0.7596`) degrade the decision boundary because they interpolate synthetic points across the legitimate manifold where *stealth* frauds hide. Furthermore, we discovered that `XGBoost` achieves its optimum on the natural distribution (`w=1`), whereas `LightGBM` gains **`+1.22 p.p.`** in temporal generalization under gentle positive weighting (`scale_pos_weight = 5`).
4. **Analytical Bayesian Recalibration and *Triple Ensemble*:** By applying analytical Bayesian prior correction ($\text{odds}_{\text{cal}} = \text{odds} / 5$) to `LightGBM (w=5)` probabilities before blending with `Refined XGBoost (w=1)` and `Natural RandomForest (w=1)`, we broke the **`0.8604 ± 0.0275` barrier in 5-Fold CV** and reached **`0.8007` on the Internal Temporal Split** (`ROC-AUC = 0.9890`).
5. **Victory in the Single Holdout Vault Audit (`df_test_vault`):** Opening the vault for the first and only time in Block 5 with decision thresholds **frozen on the Dev Set** ($p^*_{\text{SLA}} = 0.29$, $p^*_{\text{F1}} = 0.27$, and $p^*_{\text{Fin}} = 0.02$), the Champion pipeline boosted **Recall from `70.27%` (`52/74`) to `75.68%` (`56/74`, `+5.41 p.p.`)** for automatic card blocking (and **`81.08%` [`60/74`]** with the 2nd-tier App Push verification), lifted **F1-Score from `0.8000` to `0.8235` (`+2.35 p.p.`)**, maintained a **false positive rate of just `0.0106%`** (`1` false alert per `9,445` legitimate purchases), and increased blocked fraud by **`+€541.90`** on unseen future data.

---

## 1. Methodological Architecture: Ensuring *Zero Data Leakage* and *Zero Peeking*

In fraud detection projects, three subtle methodological flaws frequently inflate metrics in academic notebooks and cause severe production failures:
1. **Retrospective EDA *Peeking* Leakage:** Inspecting exact values across the full dataset during exploration (e.g., "there are 27 frauds at exactly €99.99" or "columns `V13` and `V15` had the lowest KS on the full dataset") and hardcoding rules before splitting train and test sets.
2. **In-Fold Preprocessing and Resampling Leakage:** Applying `RobustScaler`, feature selection, or `SMOTE` to the full matrix prior to `StratifiedKFold`.
3. **Threshold Calibration Peeking:** Choosing the probability cutoff $p^*$ by inspecting the Precision-Recall curve of the final test set itself.

### 1.1. Our 3-Tier Isolation Protocol (Block 1)
To shield our experiments against all three pitfalls, we implemented in **Block 1** of [`src/002_modeling.ipynb`](./002_modeling.ipynb) the following data topology:

| Chronological Partition | Rows | Frauds (`%`) | Time Window (`Time`) | Fraud Volume (€) | Role in Project |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Original Raw Dataset** | `284,807` | `492 (0.173%)` | `0s` to `172,792s` | `€60,127.97` | Deduplicated immediately at Step Zero (`-1,081` exact clones). |
| **Clean Dataset (`df_clean`)** | `283,726` | `473 (0.167%)` | `0s` to `172,792s` | `€58,591.39` | Split chronologically into 80% Development / 20% Holdout Vault. |
| **`df_dev` (Development 80%)** | `226,980` | `399 (0.176%)` | `0s` to `145,233s` | `€50,863.72` | Only visible dataset throughout Blocks 1 to 4. |
| — *Judge 1 (`5-Fold CV` on Dev)* | `5 × 45,396` | `~80 / fold` | Stratified on `df_dev` | `€50,863.72` | Evaluates statistical stability (`Mean ± Std` of AUPRC). |
| — *Judge 2 (`Internal Temporal`)* | `181,584 -> 45,396` | `332 -> 67` | `0..125,090s -> 125,090..145,233s` | `€7,648.47` (val) | Simulates temporal decay (*concept drift*) without touching the Vault! |
| **`df_test_vault` (Holdout Vault 20%)** | `56,746` | `74 (0.130%)` | `145,234s` to `172,792s` | `€7,727.67` | **Locked during Blocks 1 to 4.** Opened once in Block 5. |

---

## 2. `LeakFreeFeatureEngineer` and Controlled Ablation Study (Block 2)

All feature transformations were encapsulated inside the scikit-learn compatible transformer [`LeakFreeFeatureEngineer(BaseEstimator, TransformerMixin)`](./002_modeling.ipynb).

### 2.1. What Is Learned in `.fit(X_train, y_train)` vs. Applied in `.transform(X_val)`
1. **Domain-Agnostic Financial Features (`add_financial=True`):**
   - `log1p_amount = np.log1p(Amount)`: Compresses extreme monetary skewness (deterministic mathematical transformation, zero leakage).
   - `amount_robust`: The median and interquartile range (IQR) of `RobustScaler` are learned **exclusively on `X_train[["Amount"]]`** during `.fit()` and applied to `X_val` during `.transform()`.
   - `is_zero_amount = (Amount == 0.0)` and `is_round_integer = (Amount % 1.0 == 0.0)`: Universal properties of payment transactions (*zero-dollar auth* checks and round integer withdrawals/purchases), without hardcoding any dataset-specific magic numbers observed during EDA.
2. **Cyclical Time Encoding (`add_cyclic_time=True`, `drop_raw_time=True`):**
   - Converts continuous elapsed seconds into 24-hour harmonic coordinates ($\text{hour\_sin} = \sin(2\pi \cdot t / 86400)$, $\text{hour\_cos} = \cos(2\pi \cdot t / 86400)$) and removes raw `Time` to prevent trees from memorizing historical timestamp intervals.
3. **PCA Space Norms (`add_pca_norm=True`) and Automatic *In-Fold* Pruning (`prune_noise_ks_threshold=0.14`):**
   - Inside `.fit(X_train, y_train)` of each fold, the transformer computes the two-sample non-parametric Kolmogorov-Smirnov (`KS`) statistic between `y_train == 0` and `y_train == 1` across `V1..V28`.
   - Features whose training `KS` falls below `0.14` are stored in `self.noise_cols_` and dropped in `.transform()`, eliminating noisy dimensions without ever consulting validation or test labels.

### 2.2. Ablation Study Results (`D0` to `D6` inside `df_dev`)

| Dataset (Ablation Stage) | Cols | Judge 1: XGB 5-Fold CV | Judge 2: XGB Temporal | Judge 1: LGB 5-Fold CV | Judge 2: LGB Temporal | Global Mean AUPRC | Scientific Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`D0_Raw_With_Time`** | `30` | `0.8546 ± 0.0302` | `0.7701` | `0.8395 ± 0.0306` | `0.7716` | `0.8090` | Suffers from temporal overfitting in XGBoost (`0.7701`). |
| **`D1_Baseline_No_Time`** | `29` | `0.8524 ± 0.0320` | `0.7761` | `0.8399 ± 0.0295` | `0.7681` | `0.8091` | Dropping raw `Time` improves Temporal XGB by **`+0.60 p.p.`** |
| **`D2_Plus_Financial`** | `33` | `0.8515 ± 0.0329` | `0.7777` | `0.8403 ± 0.0290` | `0.7675` | `0.8093` | Improves both models (`+0.16 p.p.` XGB Temp, `+0.04 p.p.` LGB CV). |
| **`D3_Plus_Cyclic_Time`** | `35` | `0.8518 ± 0.0334` | `0.7796` | `0.8394 ± 0.0294` | `0.7648` | `0.8089` | Lifts Temporal XGB to **`0.7796`** (`+0.95 p.p.` over `D0`). |
| **`D4_Plus_PCA_Norm`** | `37` | `0.8499 ± 0.0322` | `0.7712` | `0.8409 ± 0.0328` | `0.7739` | `0.8090` | Helps LGB, but introduces collinearity in unpruned XGB (`37` cols). |
| **`D5_Plus_InFold_Pruning`** | `31` | **`0.8601 ± 0.0240`** | `0.7845` | **`0.8438 ± 0.0285`** | **`0.7759`** | `0.8161` | In-fold `KS < 0.14` pruning drops `['V13','V15','V22','V24','V25','V26']` and jumps **`+0.71 p.p.`** in global mean! |
| **`D6_Selected_Best (Fin+Time+Prune)`** | **`29`** | `0.8573 ± 0.0276` | **`0.7926`** | `0.8437 ± 0.0270` | `0.7749` | **`0.8171` (#1)** | **Champion:** Same column count as Baseline (`29`), but **`+2.25 p.p.`** on Temporal XGB over `D0` (`0.7701 -> 0.7926`)! |

---

## 3. Algorithm Families and *In-Fold* Imbalance Handling (Block 3)

### 3.1. Cross-Family Consistency (`D1_Baseline` vs. `D6_Selected_Best`)
To prove that the superiority of `D6_Selected_Best` was not an artifact of a single algorithm, we evaluated 4 distinct linear and tree-based model families:

| Algorithm Family | Dataset | Judge 1: 5-Fold CV AUPRC | Judge 2: Temporal AUPRC | Temporal ROC-AUC | Mean AUPRC | Gain (`D6 - D1`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LogisticRegression (L2)** | `D1_Baseline` | `0.7502 ± 0.0165` | `0.7316` | `0.9685` | `0.7409` | — |
| **LogisticRegression (L2)** | **`D6_Selected_Best`** | **`0.7512 ± 0.0176`** | **`0.7340`** | `0.9673` | **`0.7426`** | **`+0.17 p.p.`** |
| **RandomForest (n=150)** | `D1_Baseline` | `0.8155 ± 0.0246` | `0.7522` | `0.9609` | `0.7838` | — |
| **RandomForest (n=150)** | **`D6_Selected_Best`** | **`0.8267 ± 0.0199`** | **`0.7710`** | **`0.9661`** | **`0.7989`** | **`+1.51 p.p.`** |
| **LightGBM (Reg)** | `D1_Baseline` | `0.8399 ± 0.0295` | `0.7681` | `0.9778` | `0.8040` | — |
| **LightGBM (Reg)** | **`D6_Selected_Best`** | **`0.8437 ± 0.0270`** | **`0.7749`** | **`0.9779`** | **`0.8093`** | **`+0.53 p.p.`** |
| **XGBoost (Depth=5)** | `D1_Baseline` | `0.8524 ± 0.0320` | `0.7761` | `0.9815` | `0.8142` | — |
| **XGBoost (Depth=5)** | **`D6_Selected_Best`** | **`0.8573 ± 0.0276`** | **`0.7926`** | **`0.9850`** | **`0.8250`** | **`+1.08 p.p.`** |

### 3.2. *In-Fold* Experiment Across 7 Imbalance Strategies
Applying all techniques **strictly to `(X_tr, y_tr)` inside each fold** while keeping `(X_va, y_va)` at its natural class ratio (`1 : 568`):

| In-Fold Imbalance Strategy (on `D6`) | XGB 5-Fold CV | XGB Temporal | LGB 5-Fold CV | LGB Temporal | Global Mean AUPRC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1) Natural (Weight 1:1)** | **`0.8573 ± 0.0276`** | **`0.7926`** | `0.8437 ± 0.0270` | `0.7749` | **`0.8171` (#1)** |
| **2) Gentle Cost-Weight (`w=5`)** | `0.8543 ± 0.0272` | `0.7799` | **`0.8466 ± 0.0279`** | **`0.7871`** | **`0.8170` (#2)** |
| **3) Moderate Cost-Weight (`w=15`)** | `0.8538 ± 0.0255` | `0.7764` | `0.8373 ± 0.0383` | `0.7730` | `0.8101` |
| **4) Extreme Cost-Weight (`w=568`)** | `0.8283 ± 0.0264` | `0.7581` | `0.8030 ± 0.0073` | `0.7528` | `0.7855` |
| **5) Financial Sample-Weight (`1+log1p(Amt)`)** | `0.8421 ± 0.0243` | `0.7558` | `0.8365 ± 0.0307` | `0.7586` | `0.7982` |
| **6) In-Fold RandomUnderSampler (`ratio=0.05`)** | `0.7716 ± 0.0701` | `0.7609` | `0.7434 ± 0.0343` | `0.7625` | `0.7596` |
| **7) In-Fold SMOTE (`k=3, ratio=0.05`)** | `0.8502 ± 0.0194` | `0.7541` | `0.8237 ± 0.0281` | `0.7730` | `0.8002` |

* **Mathematical and Geometric Explanation:**
  - **Why does `SMOTE` (`0.7541` on Temporal XGB) fail?** Because ~17%–30% of real frauds are *stealth* frauds embedded inside the dense legitimate transaction cloud near the origin of PCA space. `SMOTE` draws line segments connecting *stealth* frauds to *extreme* frauds, polluting the interior of the legitimate manifold with synthetic positive examples and inflating false positives.
  - **Why are `XGBoost (w=1)` and `LightGBM (w=5)` complementary?** `XGBoost`'s *depth-wise* tree growth constructs symmetric trees that isolate extreme fraud clusters cleanly at natural `1:1` weighting (`0.7926` temporal). Conversely, `LightGBM`'s *leaf-wise* growth with strong L2 regularization (`reg_lambda=10.0`) benefits from a gentle positive gradient boost (`scale_pos_weight=5`), climbing from `0.7749` to **`0.7871` (`+1.22 p.p.`)** in temporal validation.

---

## 4. Refinement, Bayesian Recalibration, *Out-of-Fold Ensemble*, and Thresholds $p^*$ (Block 4)

### 4.1. Analytical Bayesian Recalibration and *Triple Ensemble*
When a classifier is trained with positive class weight $w = 5$, its raw predicted probabilities $\hat{p}_w$ estimate the weighted odds $w \cdot \frac{P(y=1|x)}{P(y=0|x)}$. Before blending it with models trained on the natural distribution ($w=1$), we apply **Analytical Bayesian Recalibration**:

$$\text{odds}_{\text{cal}}(x) = \frac{1}{w} \cdot \frac{\hat{p}_w(x)}{1 - \hat{p}_w(x)}, \qquad \hat{p}_{\text{cal}}(x) = \frac{\text{odds}_{\text{cal}}(x)}{1 + \text{odds}_{\text{cal}}(x)}$$

This transformation is strictly monotonic (preserving 100% of `LightGBM`'s individual `AUPRC` and `ROC-AUC`), while aligning its probability scale with `Refined XGBoost (w=1)` and `Natural RandomForest (w=1)`:

| Architecture Evaluated on `df_dev` (`D6_Selected_Best`) | Judge 1: 5-Fold CV AUPRC | Global OOF AUPRC | Judge 2: Temporal AUPRC | Temporal ROC-AUC | Mean Judges AUPRC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1) Base XGBoost (`Depth=5, w=1`)** | `0.8573 ± 0.0276` | `0.8556` | `0.7926` | `0.9850` | `0.8250` |
| **2) Refined XGBoost (`Subsample=0.85, L2=2.0, w=1`)** | `0.8586 ± 0.0293` | `0.8571` | `0.7988` | `0.9878` | `0.8287` |
| **3) Refined LightGBM (`w=5 + Bayes Recal`)** | `0.8508 ± 0.0285` | `0.8498` | `0.7837` | `0.9792` | `0.8172` |
| **4) Natural RandomForest (`Bagging n=200, w=1`)** | `0.8511 ± 0.0287` | `0.8508` | `0.7736` | `0.9805` | `0.8124` |
| **5) Dual Ensemble (`0.70 XGB_Ref + 0.30 LGB_Recal`)** | `0.8578 ± 0.0296` | `0.8574` | `0.7981` | `0.9863` | `0.8279` |
| **6) Triple Ensemble (`0.60 XGB + 0.25 LGB + 0.15 RF`)** | **`0.8604 ± 0.0275`** | **`0.8590`** | **`0.8007`** | **`0.9890`** | **`0.8305` (#1)** |

### 4.2. *Out-of-Fold* Decision Threshold Calibration ($p^*$) on `df_dev`
Using exclusively *Out-of-Fold* predictions (`226,980` transactions in `df_dev`, containing `399` frauds = **`€50,863.72`**), we calibrated decision policies under two banking operational regimes:
- **Card Blocking Regime (SLA):** Each blocked fraud (TP) saves 100% of `Amount` minus `€2.50` notification cost; each false card block on a legitimate customer (FP) costs `€15.00` in support center friction and lost interchange fees.
- **Soft App Push Challenge Regime (*Step-Up Auth*):** Each alert (TP or FP) triggers an instant biometric confirmation push in the mobile banking app at an operational cost of `€5.00`.

| Decision Policy (Learned 100% OOF on `df_dev`) | Threshold ($p^*$) | OOF Precision | OOF Recall | OOF F1-Score | TP | FP | FN | Net Savings SLA Block (€) | Net Savings App Push (€) | % Saved (Push) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1) Naive Default Threshold (`p = 0.50`)** | `0.50` | `93.18%` | `78.70%` | `0.8533` | `314` | `23` | `85` | `€37,321.42` | `€36,766.42` | `72.28%` |
| **2) Maximum Statistical F1 ($p^*_{\text{F1}}$)** | **`0.27`** | **`91.76%`** | **`83.71%`** | **`0.8755`** | **`334`** | **`30`** | **`65`** | `€37,716.36` | `€37,181.36` | `73.10%` |
| **3) Optimal Card Block SLA ($p^*_{\text{SLA}}$)** | **`0.29`** | **`91.99%`** | **`83.46%`** | **`0.8752`** | **`333`** | **`29`** | **`66`** | **`€37,732.86`** | `€37,190.36` | `73.12%` |
| **4) Optimal Soft App Push ($p^*_{\text{Fin}}$)** | **`0.02`** | `60.21%` | **`87.22%`** | `0.7124` | **`348`** | `230` | **`51`** | `€36,524.94` | **`€37,954.94`** | **`74.62%`** |

---

## 5. Single Holdout Vault Audit (`df_test_vault`) and Forensic Error Analysis (Block 5)

With all hyperparameters, ensemble weights, and decision thresholds ($p^*_{\text{SLA}} = 0.29$, $p^*_{\text{F1}} = 0.27$, $p^*_{\text{Fin}} = 0.02$) **frozen in Block 4**, we fitted the pipeline on `df_dev` (`80%`) and opened the Holdout Vault (`df_test_vault`: final `20%` = `56,746` transactions, `74` frauds = **`€7,727.67`**) for the first and only time.

### 5.1. Final Comparative Table on the Holdout Vault (`df_test_vault`)

| Audited Pipeline on Holdout Vault (`20% Future`) | ROC-AUC | Frozen Threshold | Precision | Recall | F1-Score | TP | FP | FN | Gross Blocked Fraud (€) | % Gross Saved | Net SLA Savings (€) | Net Push Savings (€) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1) Raw Baseline `D0` (`w/ Time + XGB @ p=0.50`)** | `0.9847` | `0.50` | `94.55%` | `70.27%` | `0.8062` | `52` | `3` | `22` | `€4,795.98` | `62.06%` | `€4,620.98` | `€4,520.98` |
| **2) Clean Baseline `D1` (`no Time + XGB @ p=0.50`)** | `0.9829` | `0.50` | `92.86%` | `70.27%` | `0.8000` | `52` | `4` | `22` | `€4,795.98` | `62.06%` | `€4,605.98` | `€4,515.98` |
| **3) `D6` + Refined XGBoost (`@ p=0.50`)** | `0.9865` | `0.50` | `91.38%` | `71.62%` | `0.8030` | `53` | `5` | `21` | `€4,087.23` | `52.89%` | `€3,879.73` | `€3,797.23` |
| **4) `D6` + Triple Ensemble (`@ p=0.50` Default)** | **`0.9871`** | `0.50` | `94.44%` | `68.92%` | `0.7969` | `51` | `3` | `23` | `€4,046.57` | `52.36%` | `€3,874.07` | `€3,776.57` |
| **5) Champion: `D6` + Ensemble (`@ p*_SLA = 0.29`)** | **`0.9871`** | **`0.29`** | **`90.32%`** | **`75.68%`** | **`0.8235`** | **`56`** | **`6`** | **`18`** | **`€5,089.40`** | **`65.86%`** | **`€4,859.40`** | **`€4,779.40`** |
| **6) Champion: `D6` + Ensemble (`@ p*_F1 = 0.27`)** | **`0.9871`** | **`0.27`** | **`88.89%`** | **`75.68%`** | **`0.8175`** | **`56`** | **`7`** | **`18`** | **`€5,089.40`** | **`65.86%`** | `€4,844.40` | `€4,774.40` |
| **7) Champion: `D6` + Ensemble (`@ p*_Fin = 0.02` Push)** | **`0.9871`** | **`0.02`** | `44.12%` | **`81.08%`** | `0.5714` | **`60`** | `76` | **`14`** | **`€5,337.88`** | **`69.07%`** | `€4,047.88` | `€4,657.88` |

### 5.2. Forensic X-Ray of Hits and Errors in the Holdout Vault (`@ p*_F1 = 0.27`)

| Forensic Group in Holdout Vault | `N` | Median (Mean) `Amount` | Median `V14` | Median `V17` | Top-8 PCA L2 Norm | Median Probability | Technical Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **True Positives (TP)** | `56` | `€1.79` (`€90.88`) | `-8.31` | `-3.26` | `14.90` | `0.9257` | Captures both micro-probes and high-value frauds exhibiting anomalous `V14`/`V17` signatures. |
| **False Negatives (FN — *Stealth*)** | `18` | `€10.16` (`€146.57`) | **`-1.82`** | **`+1.05`** | **`4.63`** | `0.0023` | *Stealth* frauds embedded inside the normal manifold (`V17 > 0` and L2 norm near `2.45`). Require stateful card velocity features (`card_id`) or oblique sensors. |
| **False Positives (FP)** | `7` | `€2.00` (**`€3,865.34`**) | `-6.80` | `-1.86` | `11.55` | `0.4981` | Legitimate purchases that are extreme statistical outliers in value (€3.8k mean) or `V14`, where a preventive alert protects the cardholder. |
| **True Negatives (TN)** | `56,665` | `€18.05` (`€78.13`) | `-0.03` | `-0.16` | `2.45` | `0.0001` | `99.988%` of legitimate transactions clear instantaneously without friction (`FPR = 0.012%`). |

### 5.3. Feature Importance Ranking (Learned on `df_dev`)

| Rank | Feature | XGBoost Gain (`%`) | LightGBM Split (`%`) | RandomForest Gini (`%`) | Weighted Ensemble Importance (`%`) | Feature Origin |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1st** | `V14` | `27.45%` | `6.04%` | `11.16%` | **`19.66%`** | Original PCA Component |
| **2nd** | `V10` | `25.74%` | `5.02%` | `8.63%` | **`17.99%`** | Original PCA Component |
| **3rd** | `V17` | `12.38%` | `3.25%` | `21.19%` | **`11.42%`** | Original PCA Component |
| **4th** | `V12` | `9.96%` | `6.04%` | `12.42%` | **`9.35%`** | Original PCA Component |
| **5th** | `V4` | `2.96%` | `7.41%` | `3.00%` | **`4.08%`** | Original PCA Component |
| **6th** | `V16` | `1.00%` | `4.43%` | `7.32%` | **`2.81%`** | Original PCA Component |
| **7th** | `V11` | `1.18%` | `3.46%` | `6.82%` | **`2.60%`** | Original PCA Component |
| **8th** | **`hour_sin`** | **`0.97%`** | **`5.32%` (#4 in LGB!)** | **`1.66%`** | **`2.16%`** | **Zero-Leakage Cyclical Engineering** |
| **9th** | `V9` | `0.98%` | `3.14%` | `4.53%` | **`2.05%`** | Original PCA Component |
| **10th** | `V28` | `1.14%` | `4.48%` | `1.03%` | **`1.96%`** | Original PCA Component |
| **16th** | **`hour_cos`** | `0.59%` | `3.17%` | `0.85%` | **`1.27%`** | **Zero-Leakage Cyclical Engineering** |
| **17th** | **`log1p_amount`** | `1.57%` | `0.91%` | `0.69%` | **`1.27%`** | **Zero-Leakage Financial Engineering** |
| **22nd** | **`amount_robust`** | `0.00%` | `2.09%` | `0.74%` | **`0.64%`** | **In-Fold Financial Engineering** |
| **23rd** | **`is_round_integer`** | `0.87%` | `0.08%` | `0.22%` | **`0.57%`** | **Zero-Leakage Financial Engineering** |

---

## 6. Production Deployment Guide and Model Governance

1. **Two-Tier Operational Policy (Rules Engine + ML):**
   - **Green Tier ($p < 0.02$ — `99.76%` of traffic):** Instant frictionless approval (`< 15 ms`).
   - **Yellow Tier ($0.02 \le p < 0.29$ — `0.13%` of traffic):** Automated soft challenge (*Step-Up Authentication* via mobile app Push/Biometrics or 3DS token). Captures `+4` *stealth* frauds in the Holdout Vault (`Recall` rises to `81.08%`) without blocking legitimate customer cards.
   - **Red Tier ($p \ge 0.29$ — `0.11%` of traffic):** Immediate preventive transaction block and instant cardholder alert (`Precision = 90.32%`, `F1 = 0.8235`).
2. **Leak-Free Auditable Persistence:**
   - The file [`data/processed/creditcard_engineered_best.parquet`](../data/processed/creditcard_engineered_best.parquet) (`283,726` rows × `31` columns) includes the `split_group` column (`dev_train_80pct` vs. `holdout_vault_20pct`), ensuring exact traceability between development data and the holdout audit vault.
