\# Multi-Modal XAI Pneumonia DSR Framework



Open-source research prototype developed for the DTS481 Paper 3 Design Science Research (DSR) study.



\## Research Artefact



The prototype implements a multi-modal pneumonia classification framework that combines:



1\. \*\*CXR image modality\*\* — transfer learning using DenseNet-121.

2\. \*\*Structured metadata modality\*\* — age, sex and view position from the NIH ChestX-ray14 dataset.

3\. \*\*Multi-modal fusion\*\* — image and metadata embeddings are combined before classification.

4\. \*\*XAI\*\* — Grad-CAM is used to provide visual explanations of model predictions.

5\. \*\*HITL\*\* — a Streamlit review interface allows a human reviewer to accept or reject a prediction and provide a corrected label.

6\. \*\*Evaluation\*\* — diagnostic performance is evaluated using accuracy, precision, recall, specificity, F1-score, ROC-AUC and PR-AUC.



The system is intentionally a research prototype and is \*\*not a clinical diagnostic device\*\*.



\## Dataset



The primary dataset used by the prototype is NIH ChestX-ray14. It contains 112,120 frontal chest X-rays from 30,805 patients and includes metadata such as age, sex and view position.



The dataset is publicly available and de-identified.



Use the official NIH dataset source:



https://nihcc.app.box.com/v/ChestXray-NIHCC



Expected layout:



data/

&#x20; raw/

&#x20;   images/

&#x20;     images-001/

&#x20;     images-002/

&#x20;     ...

&#x20;   Data\_Entry\_2017\_v2020.csv

&#x20;   train\_val\_list.txt

&#x20;   test\_list.txt



The implementation uses:



\- \*\*Pneumonia\*\* as the positive class

\- \*\*No Finding\*\* as the negative class



This avoids treating unrelated diseases as a clean negative class.



\## Research Design



The artefact follows a Design Science Research approach and evaluates the framework through four experimental stages.



| Experiment | Purpose |

|---|---|

| \*\*E1\*\* | Image-only DenseNet-121 baseline |

| \*\*E2\*\* | Multi-modal DenseNet-121 using image and metadata |

| \*\*E3\*\* | Grad-CAM explainability analysis |

| \*\*E4\*\* | Human-in-the-Loop review and correction |



The experiments progressively evaluate classification performance, explainability and human oversight.



\## Experimental Results



The final test-set results were:



| Experiment | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC |

|---|---:|---:|---:|---:|---:|---:|---:|

| E1 Image-only | 0.9098 | 0.0983 | 0.3289 | 0.9244 | 0.1514 | 0.7775 | 0.0862 |

| E2 Multi-modal | 0.9239 | 0.1113 | 0.3026 | 0.9394 | 0.1627 | 0.7815 | 0.0858 |



The multi-modal model improved overall accuracy, specificity, precision and F1-score compared with the image-only baseline, while recall decreased slightly.



The E3 Grad-CAM experiment evaluated four representative cases:



\- True Positive (TP)

\- True Negative (TN)

\- False Positive (FP)

\- False Negative (FN)



The E4 HITL experiment demonstrated that the reviewer correctly identified and corrected both model errors in the four-case evaluation.



Detailed metrics and visual outputs are available in the `outputs/` directory.



\## Project Structure



```text

src/

\&#x20; config.py

\&#x20; dataset.py

\&#x20; models/

\&#x20; train.py

\&#x20; evaluate\\\_image\\\_only.py

\&#x20; evaluate\\\_multimodal.py

\&#x20; select\\\_xai\\\_cases.py

\&#x20; explain.py

\&#x20; analyze\\\_xai.py

\&#x20; analyze\\\_hitl.py

\&#x20; summarize\\\_results.py



app/

\&#x20; streamlit\\\_app.py



scripts/

\&#x20; prepare\\\_nih.py

\&#x20; apply\\\_feedback.py



outputs/

\&#x20; figures/

\&#x20; metrics/

\&#x20; explanations/

\&#x20; feedback/

Generated model checkpoints, raw datasets and raw HITL feedback are excluded from version control.



Setup

1\. Create a virtual environment



Windows PowerShell:



py -3.12 -m venv .venv

.\\.venv\\Scripts\\Activate.ps1



If PowerShell blocks activation:



Set-ExecutionPolicy -Scope Process Bypass

2\. Install PyTorch



Install the appropriate PyTorch version for your hardware using the official PyTorch installation instructions:



https://pytorch.org/get-started/locally/



3\. Install the remaining dependencies

pip install -r requirements.txt

4\. Prepare the NIH dataset

python scripts/prepare\_nih.py

5\. Train the models



For a small initial test:



python -m src.train --epochs 1 --max-samples 2000



For the full experiment, use the documented configuration and dataset.



6\. Evaluate the models



Run the image-only and multi-modal evaluation scripts according to the experimental configuration.



7\. Generate Grad-CAM explanations

python -m src.explain

8\. Launch the HITL interface

streamlit run app/streamlit\_app.py

Research Controls



The experimental design incorporates the following controls:



Patient-level dataset splitting.

A separate test set for final evaluation.

Fixed experimental configurations.

Recorded evaluation metrics.

Explicit handling of class imbalance.

Representative TP, TN, FP and FN cases for XAI analysis.

Human review of selected model predictions.

No clinical claims are made from the prototype results.

DSR Mapping

Problem Identification and Motivation



The research addresses limitations in automated pneumonia diagnosis related to diagnostic performance, limited use of structured metadata, lack of prediction transparency and limited human involvement.



Objectives of the Solution



The artefact aims to investigate whether combining image and metadata information can improve pneumonia classification while providing interpretable predictions and supporting human oversight.



Design and Development



The framework was implemented using Python, PyTorch, DenseNet-121, Grad-CAM and Streamlit.



Demonstration



The prototype demonstrates automated pneumonia prediction, visual explanation generation and human review through the implemented software components.



Evaluation



The artefact was evaluated through four experiments:



E1 — image-only baseline

E2 — multi-modal classification

E3 — Grad-CAM XAI

E4 — Human-in-the-Loop review

Communication



The resulting framework, source code, experimental outputs and research findings are communicated through this repository and the accompanying DTS481 research paper.



Limitations



The prototype is intended for research purposes only. It does not constitute a clinically validated diagnostic system.



The evaluation uses a limited set of metadata fields available in the NIH ChestX-ray14 dataset and does not represent a complete clinical record.



The XAI evaluation uses Grad-CAM and a small number of representative cases rather than a large-scale clinical explanation study.



The HITL experiment uses a small controlled review set and therefore cannot establish clinical effectiveness or general human-reader performance.



External-dataset generalisability was not evaluated in the current implementation.



License



The framework code is released under the MIT License.



The NIH ChestX-ray14 dataset remains subject to the terms and conditions of its dataset provider. The repository license does not grant redistribution rights for the dataset.



E7 — ResNet-50 single-model evaluation



E7 adds a metadata-fused ResNet-50 comparison while preserving E1-E6. It uses the same processed manifest, patient split, image size, age/sex/view metadata, class weighting and core training policy as E2. Baseline preprocessing is fixed for E7; CLAHE and perturbation robustness remain E9 scope.



The E7 protocol is defined in `config_e7.yaml`. Its repository-relative `data.image_root` resolves the unchanged manifest filenames against the current `data/raw/images/images-###` folders, so a moved checkout does not depend on stale absolute manifest paths. Its outputs use E7-specific filenames and do not overwrite the retained E1-E6 checkpoints or metrics.



Run the contract checks before training:

`python -m unittest discover -s tests -p "test_e7_contracts.py"`



Training is deliberately user-started because a full run may take hours:

`python -m src.train_e7 --config config_e7.yaml`



After training, validation inference selects and locks the classification threshold using validation F1 only:

`python -m src.evaluate_e7 --config config_e7.yaml --split val`



Only after the validation threshold artefact has been reviewed should the held-out test evaluation be run:

`python -m src.evaluate_e7 --config config_e7.yaml --split test`



E7 retains per-image validation and test probabilities under `outputs/predictions/` so later paired analysis and the E8 DenseNet-121 plus ResNet-50 ensemble can use aligned predictions. E7 remains a binary, single-dataset research experiment and is not clinical validation.



E8 — DenseNet-121 plus ResNet-50 ensemble evaluation



E8 performs inference only; it does not train or fine-tune either model. It regenerates aligned E2 DenseNet-121 probabilities and combines them with the retained E7 ResNet-50 probabilities using the pre-specified rule:

`ensemble probability = (0.5 * DenseNet probability) + (0.5 * ResNet probability)`



Run the non-inference contract checks first:

`py -m unittest discover -s tests -p "test_e8_contracts.py" -v`



The validation stage reproduces the historical E2 validation evidence, aligns both models by unique image ID and target, and locks the E8 threshold using validation F1 only:

`py -m src.evaluate_e8 --config config_e8.yaml --stage val`



Only after reviewing the validation lock should the test stage be run once:

`py -m src.evaluate_e8 --config config_e8.yaml --stage test`



All E8 outputs use isolated `e8_*` filenames. The evaluator refuses existing outputs and refuses test execution without a hash-matched validation lock. E2 retains threshold 0.75, E7 retains threshold 0.85, and E8 uses its own validation-selected threshold. The comparison remains binary, retrospective, single-dataset research evidence and does not establish multi-class or clinical performance.



E9 — Baseline-versus-CLAHE robustness and feature stability



E9 compares the existing baseline preprocessing pipeline with CLAHE across DenseNet-121, ResNet-50 and their equal-weight probability ensemble. In this experiment, “baseline” means the existing resize, tensor conversion and ImageNet normalization pipeline without CLAHE; it does not mean untouched detector data. ICEA is not implemented and remains an explicit limitation.



The retained E2 checkpoint supplies the baseline DenseNet arm. E6-A is verified as an exact duplicate of E2 and is not retrained. The retained E6-B checkpoint supplies the CLAHE DenseNet arm, and E7 supplies the baseline ResNet arm. Only the matching CLAHE ResNet-50 requires new five-epoch training.



Run the E9 contract checks first:

`py -m unittest discover -s tests -p "test_e9_contracts.py" -v`



Training is deliberately user-started because the CPU run may take many hours:

`py -m src.train_e9 --config config_e9.yaml`



After training, regenerate the retained E6-B validation result and lock separate CLAHE thresholds for DenseNet, ResNet and the equal-weight ensemble using validation F1 only:

`py -m src.evaluate_e9 --config config_e9.yaml --stage val`



Run the clean held-out test condition first. This must reproduce the retained E2, E7 and E8 baseline probabilities before it creates the clean feature reference:

`py -m src.evaluate_e9 --config config_e9.yaml --stage condition --condition clean`



Then run each deterministic perturbation separately so completion and evidence are recoverable:

`py -m src.evaluate_e9 --config config_e9.yaml --stage condition --condition brightness_low`

`py -m src.evaluate_e9 --config config_e9.yaml --stage condition --condition brightness_high`

`py -m src.evaluate_e9 --config config_e9.yaml --stage condition --condition contrast_low`

`py -m src.evaluate_e9 --config config_e9.yaml --stage condition --condition contrast_high`

`py -m src.evaluate_e9 --config config_e9.yaml --stage condition --condition gamma_low`

`py -m src.evaluate_e9 --config config_e9.yaml --stage condition --condition gamma_high`

`py -m src.evaluate_e9 --config config_e9.yaml --stage condition --condition gaussian_noise`

`py -m src.evaluate_e9 --config config_e9.yaml --stage condition --condition gaussian_blur`



After all nine conditions are complete and their hashes validate, calculate point estimates and 1,000 paired, stratified bootstrap intervals:

`py -m src.evaluate_e9 --config config_e9.yaml --stage summarize`



Perturbations are applied deterministically to the source RGB image before the baseline or CLAHE arm. Feature stability uses the 256-dimensional post-ReLU `image_projection` representation, before metadata fusion. Ensemble feature stability is reported only as the mean of the two component-model stability values and is labelled as a proxy, because probability fusion has no learned ensemble feature vector. Thresholds are locked on clean validation data and never retuned on perturbed test conditions.



E9 remains a retrospective, binary, single-dataset robustness experiment. It does not establish robustness to external institutions, scanners or populations, does not compare ICEA, and does not constitute clinical validation.


