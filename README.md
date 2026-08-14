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


