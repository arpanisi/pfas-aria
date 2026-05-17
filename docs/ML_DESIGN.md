# Statistical and ML Design

PFAS-ARIA is an evidence-building system for experimental PFAS degradation data. Its goal is not to train a single predictive model. Its goal is to take a structured experimental dataset, test many plausible predictor-outcome relationships, rank the strongest statistical signals, and then check whether those signals are consistent with relevant literature.

The system should be read as a hypothesis screening and evidence triage workflow. It is designed to help a scientist decide what relationships deserve closer mechanistic interpretation, replication, or confirmatory modeling.

---

## 1. Statistical Goal

The central question is:

> Which experimental conditions, chemical descriptors, time variables, or treatment settings are most strongly associated with PFAS degradation outcomes, and are those associations supported by prior literature?

For each experimental regime, the system searches over candidate relationships of the form:

```text
outcome = f(selected predictors) + error
```

Examples of outcomes may include final concentration, removal fraction, fluoride release, degradation rate, or other measured response variables. Predictors may include treatment type, time, energy input, concentration, pH, compound class, reactor settings, additives, or encoded categorical factors.

The output is a ranked list of candidate hypotheses, not a causal proof. Each candidate is accompanied by:

- estimated effect sizes or coefficients
- p-values or model-derived importance measures where available
- fit statistics such as R2 and adjusted R2
- diagnostic checks
- literature matches

---

## 2. Data Framing

The system assumes the uploaded dataset contains repeated experimental observations with some columns acting as inputs and others as measured outputs. When the workbook identifies input and output roles, those roles constrain the search space. This prevents the system from treating every variable as both a predictor and an outcome.

Before modeling, the data are normalized into a tabular form:

- column names are standardized
- eligible categorical values are encoded numerically
- missing values are preserved unless a specific modeling step handles them
- experimental/time identifiers are retained when available
- regime labels are used to analyze distinct experimental systems separately

The regime-level framing is important. PFAS degradation experiments often combine heterogeneous systems: different compounds, reactors, treatments, measurement schedules, or operating conditions. A global model across all rows can hide regime-specific behavior. The screening workflow therefore focuses on one experimental regime at a time when possible.

### Current Data Treatment

The active upload and screening workflow is conservative about missing data. Blank spreadsheet cells and placeholder values such as `-`, `N/A`, or `ND` are treated as missing values. They are not silently converted to zero. In the upload preview, missing values are displayed as blank cells because the preview is showing the cleaned tabular data, not an imputed modeling matrix.

This distinction matters statistically. A zero can mean a true measured absence, while a blank cell may mean not measured, not applicable, below detection, or unavailable. Treating all missing cells as zero would impose a strong assumption before the model has any evidence for it.

What is currently implemented in the active framing path:

- unified Excel sheets are parsed using the declared input/output layout
- column names are normalized
- fully empty rows and fully empty columns are removed
- categorical columns are label-encoded only when they are genuinely categorical
- numeric-looking columns with placeholder strings are coerced to numeric, with placeholders becoming missing values
- input and output column roles are preserved for screening
- missing cells remain missing through upload preview and screening preparation

Some columns can appear empty in the first preview rows even when they contain values later in the dataset. The preview is a sample of the cleaned table, not a completeness report for the full dataset.

### Available But Not Fully Used

The codebase also contains a more aggressive experimental ETL path. That path can validate data, transform features, drop columns with high missingness, impute missing numeric feature values using the median by default, create log-transformed variables for skewed predictors, cache processed data, and return a modeling-ready data bundle.

That ETL path is not currently the main upload/screening path. It exists as implementation groundwork for a stricter preprocessing mode, but the current screening workflow intentionally avoids automatic imputation at upload time.

What could be added deliberately in a future data-framing step:

- a missingness report by input/output role
- explicit distinction between true zero, below-detection, not measured, and not applicable
- optional median or model-based imputation for selected numeric predictors
- sensitivity checks comparing complete-case models against imputed-data models
- rules for dropping columns above a predefined missingness threshold
- per-regime missingness checks, since a column may be absent in one experimental regime but meaningful in another
- explicit user-visible notes when a preview column is blank only because the first sampled rows are missing

Any imputation should be treated as a modeling assumption, not as basic ingestion. For screening, the safer default is to preserve missingness first, then test whether conclusions are robust to reasonable preprocessing choices.

---

## 3. Candidate Hypothesis Screening

The screening stage fits many shallow, interpretable candidate models rather than one large model. This is intentional.

The system is usually trying to answer:

- Which small sets of predictors explain meaningful variation in an outcome?
- Which predictors remain important after basic controls are included?
- Which effects are stable enough to survive diagnostics?
- Which statistically strong relationships also resemble mechanisms described in the literature?

Candidate predictor sets are generated from the available input columns and evaluated against output columns. Each candidate is scored using a combination of fit quality, diagnostics, and literature similarity.

### Primary Model

The default screening model is linear regression:

```text
y = beta_0 + beta_1 x_1 + ... + beta_k x_k + epsilon
```

Linear models are used because the first objective is interpretability. Coefficients, directions, uncertainty, and variable-level summaries are more useful at the screening stage than a black-box prediction score alone.

When ordinary least squares is unstable or poorly conditioned, the system can fall back to ridge regression for a more stable numerical fit. Ridge estimates are used pragmatically for ranking and robustness; they should not be interpreted as classical OLS inference.

### Panel-Aware Modeling

When the data contain repeated measurements by experiment, compound, treatment group, or time, the system can use panel-style fixed-effect models. The purpose is to separate within-entity variation from between-entity differences.

This matters when repeated observations are not independent. For example, time-course degradation measurements from the same experiment should not be treated the same as unrelated cross-sectional observations.

Panel models are preferred when:

- an entity identifier is available or can be derived
- a time variable exists
- there are enough repeated observations
- predictors vary within entities

If those conditions are not met, the system falls back to simpler models.

---

## 4. Models Included

PFAS-ARIA includes several model families, but they do not all play the same role. Some are the primary models used for the screening UI; some are available in the fuller agentic hypothesis pipeline; others are auxiliary models used for robustness comparisons.

### Screening Models

These are the models most directly involved in the screening-first workflow.


| Model                     | Purpose                                                                                 | Main statistical output                                            | Robustness checks                                                                                                    |
| ------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| Ordinary least squares    | Default interpretable screening model.                                                  | Coefficients, p-values, R2, adjusted R2, residuals.                | Heteroscedasticity, autocorrelation, residual normality, functional form, VIF, condition number, Cook's distance.    |
| Ridge regression fallback | Stabilizes estimation when OLS is singular, poorly conditioned, or numerically fragile. | Penalized coefficients and fit score.                              | Treated as a fallback with an interpretability penalty; classical OLS p-values and assumption tests are not used.    |
| Panel fixed effects       | Handles repeated measurements within experiments/entities when panel structure exists.  | Within-entity effect estimates, within/between R2 where available. | Intraclass correlation, stationarity fraction, within-R2, entity-count adequacy, joint significance where available. |


The default ranking prefers interpretable OLS or panel results when they pass basic quality checks. Ridge is useful for numerical stability, but it is not treated as equally inferential because shrinkage changes the usual coefficient uncertainty interpretation.

### Full Hypothesis Pipeline Models

The broader agentic pipeline can fit additional model families.


| Model                              | Purpose                                                                      | Main statistical output                                                 | Robustness checks                                                                                                       |
| ---------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| OLS                                | Baseline parametric association model.                                       | Coefficients, p-values, residuals, fitted values, R2.                   | VIF, Shapiro-Wilk residual normality, Breusch-Pagan homoscedasticity, cross-validated R2, regime ANOVA when applicable. |
| Fixed effects panel model          | Controls for unobserved time-invariant entity differences.                   | Within-entity estimates and panel fit statistics.                       | Panel suitability, repeated-observation adequacy, fallback to OLS if panel structure is invalid or fit fails.           |
| Random effects panel model         | Models entity-level variation when random effects assumptions are plausible. | Random-effects estimates and panel fit statistics.                      | Same panel-index checks as fixed effects; fallback to OLS on fit failure.                                               |
| LASSO                              | Sparse variable selection in higher-dimensional predictor sets.              | Nonzero coefficients, selected variables, cross-validated penalty.      | Cross-validation, sparsity ratio, coefficient shrinkage awareness; p-values are not treated as classical inference.     |
| Gradient boosting regression       | Nonlinear predictive benchmark and feature-importance comparison.            | Feature importances, in-sample fit.                                     | Importance-based significance only; validation relies on predictive checks rather than linear-model assumptions.        |
| XGBoost regression                 | Nonlinear tree model with native missing-value handling.                     | Gain/importance scores, R2-like fit summaries.                          | Cross-validated R2 and generalization gap; negative CV R2 or large train-test gap is penalized.                         |
| Two-stage kinetics/treatment model | Separates time-course kinetics from between-treatment effects.               | Stage 1 within-entity kinetic fit; stage 2 treatment-level association. | Stage 1 within-R2, stage 2 R2, entity-count adequacy, underpowered stage warnings.                                      |


The full pipeline uses these models to test structured hypotheses, not to automatically crown the most complex model as best. A nonlinear model with better fit but weak interpretability may be useful as a robustness signal without replacing the primary scientific interpretation.

### Auxiliary Screening and Comparison Models

Some modules also include broader automated screening or diagnostic comparisons:


| Model                  | Role                                                      |
| ---------------------- | --------------------------------------------------------- |
| Elastic net            | Penalized linear comparison balancing L1 and L2 behavior. |
| Random forest          | Nonlinear ensemble comparison for predictive signal.      |
| RidgeCV                | Cross-validated ridge comparison in extended diagnostics. |
| Mixture of regressions | Experimental regime/latent-class style research module.   |
| Grouped fixed effects  | Experimental grouped-panel research module.               |


These are not the main interpretive layer of the current results page. They are useful for sensitivity analysis, pressure tests, and future model-selection work.

---

## 5. Robustness Testing by Model Family

Robustness is tested differently depending on the model. The system does not pretend that one diagnostic suite applies equally to OLS, penalized regression, panel models, and tree ensembles.

### OLS Robustness

OLS receives the most complete classical diagnostic suite because its assumptions are explicit and testable.

The system checks:

- **Multicollinearity:** variance inflation factor; high VIF indicates unstable coefficient estimates.
- **Heteroscedasticity:** Breusch-Pagan style checks; failure means standard errors and p-values may be unreliable.
- **Residual normality:** Shapiro-Wilk in the full pipeline and Jarque-Bera in screening diagnostics.
- **Autocorrelation:** Durbin-Watson in screening diagnostics.
- **Functional form:** RESET-style test for omitted nonlinear structure.
- **Influential observations:** Cook's distance.
- **Numerical conditioning:** condition number.
- **Out-of-sample signal:** K-fold cross-validated R2 in the full validation path.

A strong OLS candidate is one whose effect direction is interpretable, whose adjusted R2 is meaningful, and whose diagnostics do not show severe fragility.

### Ridge Robustness

Ridge is used when OLS is unstable. Its robustness value is numerical stability under multicollinearity or near-singular design matrices.

However, ridge changes the estimand by shrinking coefficients. Therefore:

- ridge coefficients are useful for ranking and directional comparison
- ridge p-values are not treated as valid classical inference
- ridge receives an interpretability penalty in diagnostics
- a ridge-only result should usually be followed by a clearer confirmatory model

### Panel Model Robustness

Panel models are robust only if the data actually support a panel interpretation.

The system checks:

- whether entity and time structure exist
- whether there are enough repeated observations
- whether predictors vary within entities
- intraclass correlation, to assess whether entity effects matter
- stationarity fraction for within-entity series where applicable
- within-R2 and between-R2
- number of entities, because too few entities make panel estimates fragile

If panel structure is inadequate or the panel model fails, the system falls back to simpler regression rather than forcing a panel interpretation.

### LASSO Robustness

LASSO is tested as a sparse-selection model, not as a classical inference model.

The system checks:

- cross-validated fit
- number of nonzero coefficients
- sparsity ratio
- whether selected variables remain scientifically interpretable

Because LASSO performs variable selection and shrinkage simultaneously, its selected coefficients should be treated as screening evidence. They are not a substitute for post-selection inference.

### Gradient Boosting and XGBoost Robustness

Tree ensembles are used to detect nonlinear predictive structure and compare variable importance against linear models.

The system checks:

- cross-validated R2 where available
- generalization gap between in-sample and cross-validated performance
- whether feature importance is concentrated in plausible variables
- whether the model appears to overfit

For XGBoost, a large generalization gap is penalized. Negative cross-validated R2 is a warning that the model may not generalize beyond the fitted sample.

Tree-based importances are not p-values. They indicate predictive contribution, not a signed mechanistic effect.

### Two-Stage Model Robustness

The two-stage model is used when the data have a kinetic structure that should be separated from treatment-level differences.

The robustness checks are:

- Stage 1 within-R2: whether time explains meaningful within-entity degradation behavior.
- Stage 2 R2: whether treatment-level predictors explain the derived kinetic summaries.
- Entity count: whether enough independent experimental entities exist for stage 2.

If there are too few entities, the model can still run, but the treatment-stage interpretation is marked as underpowered.

### Regime Robustness

Across model families, regime structure is treated as a major robustness concern.

The system asks:

- does the relationship persist within a regime rather than only globally?
- do outcomes differ meaningfully across regimes?
- does a predictor matter only because it proxies for regime membership?
- are there enough observations inside each regime?

Regime-specific screening is therefore preferred for heterogeneous experimental datasets.

---

## 6. Ranking Evidence

A candidate relationship is not ranked by R2 alone. The ranking combines three broad evidence classes:

```text
statistical fit x diagnostic credibility
```

### Fit Quality

Fit quality includes R2, adjusted R2, observation count, coefficient estimates, and significance information where available. Adjusted R2 is important because the system compares models with different numbers of predictors.

High R2 alone is not sufficient. A model can fit well because it is over-specified, unstable, or dominated by a small number of observations.

### Diagnostic Credibility

Diagnostics are used to penalize candidates that look statistically fragile. Depending on the model family and data shape, checks may include:

- multicollinearity
- residual behavior
- heteroscedasticity
- influential observations
- condition number
- panel structure quality
- within-entity versus between-entity explanatory power
- basic cross-validated performance

The diagnostic score is not a formal acceptance theorem. It is a practical ranking signal that helps keep obviously brittle candidates from dominating the results.

---

## 7. Interpretation of Coefficients

For linear models, coefficient signs are interpreted as directional associations conditional on the included predictors:

- positive coefficient: higher predictor values are associated with higher outcome values
- negative coefficient: higher predictor values are associated with lower outcome values

The scientific meaning depends on the outcome. For example, a positive effect on fluoride release may suggest stronger defluorination, while a positive effect on final PFAS concentration may indicate poorer degradation.

The system does not assume that all positive coefficients are beneficial. Outcome semantics must be interpreted by the scientist.

Categorical variables may be encoded numerically during upload cleanup. Encoded coefficients should be interpreted cautiously unless the encoding map is inspected. A categorical code is a modeling convenience, not necessarily an ordinal physical scale.

---

## 8. Validation and Robustness

The full hypothesis pipeline applies post-fit validation tests to fitted model results. These include multicollinearity checks, residual normality, heteroscedasticity checks, regime-level ANOVA, cross-validated R2, and effect-size summaries.

The screening-first workflow uses a lighter but broader diagnostic strategy because it evaluates many candidate models. Its purpose is to rank and triage, not to certify final inference.

A candidate should be considered stronger when:

- the direction and magnitude are interpretable
- the relationship is not driven by severe multicollinearity
- residual diagnostics are not pathological
- the fit is not dependent on a tiny sample
- similar signals appear across related outcomes or regimes
- literature context is mechanistically plausible

A candidate should be treated cautiously when:

- the sample size is small
- predictors are highly collinear
- categorical encodings dominate the model
- fit is high but diagnostics are poor
- literature similarity is weak or only topical
- the relationship is known to be nonlinear but only a linear approximation was fit

---

## 9. What the System Does Not Claim

PFAS-ARIA does not claim that a screened association is causal.

It does not claim that embedding similarity proves literature support.

It does not claim that a single high-R2 model identifies a degradation mechanism.

It does not claim that encoded categorical variables are physical continuous quantities.

It does not train or fine-tune embedding models on the user's dataset.

The intended claim is narrower:

> Given this dataset and literature corpus, these candidate relationships are statistically notable, diagnostically more credible than alternatives, and more or less aligned with relevant scientific text.

---

## 10. How to Read the Output

The ranked hypotheses should be treated as a shortlist for expert review.

For each candidate, inspect:

1. Sample size and regime definition.
2. Outcome semantics.
3. Included predictors and possible omitted variables.
4. Coefficient signs and magnitudes.
5. p-values or importance measures, depending on model family.
6. R2 and adjusted R2.
7. Diagnostic warnings.
8. Whether literature matches are mechanistic or merely topical.
9. Whether the relationship should be tested with a more specific model.

Good next steps may include:

- fitting a pre-specified confirmatory model
- testing nonlinear transformations
- adding interaction terms suggested by chemistry
- comparing within-regime and across-regime effects
- using mixed-effects models when repeated-measure structure is strong
- validating against held-out experiments
- checking whether literature-supported mechanisms hold for the specific PFAS class in the dataset

---

## 11. Practical Modeling Philosophy

The system favors transparent, inspectable models at the screening stage. That is a deliberate choice.

PFAS degradation datasets are often small, heterogeneous, and experimentally structured. In that setting, a large predictive model may produce a better black-box score while offering less scientific value. The workflow therefore emphasizes:

- many simple candidate tests
- regime-specific analysis
- explicit diagnostics
- literature grounding
- clear separation between computed evidence and downstream explanation

The result is not an automated final conclusion. It is a structured evidence map that helps scientists decide where deeper statistical and mechanistic work should focus.
