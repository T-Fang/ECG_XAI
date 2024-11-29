# ECG_XAI

This repository contains the implementation for my Final Year Project at the National University of Singapore (advisor: [Prof. Brian Y. Lim](https://scholar.google.com/citations?user=_bza0AoAAAAJ)). The aim of this project is to develop an explainable AI (XAI) framework that aligns machine learning predictions with clinical reasoning in electrocardiogram (ECG) diagnosis, providing clinically relevant explanations to support cardiologists' decision-making.

Previous literatures on applying AI or machine learning to ECG diagnosis often focus on achieving high prediction accuracy. Although some papers do utilize some post-hoc XAI techniques to highlight parts of ECG that are most relevant to the prediction, these highlights often fail to provide insights aligned with cardiologists' clinical reasoning, which relies on identifying specific ECG patterns indicative of heart conditions.

To address this, I designed an inherently explainable AI framework that replicates cardiologists’ decision-making processes by incorporating logical rules used in ECG diagnoses. This framework retains the advantages of machine learning in detecting nuanced details while aligning its outputs with clinical reasoning. Furthermore, my framework can generate detailed diagnostic reports that explain predictions, such as pointing out relevant ECG patterns, making the results more interpretable and trustworthy for medical professionals.

<p align="center">
    <img src="https://github.com/user-attachments/assets/66c8999c-e373-4027-9f30-0e50c384c77d" alt="">
    Fig1: The 12-lead ECG plot of a patient with anterior myocardial infarction (AMI). The ECG XAI model derived from the proposed framework can provide detailed explanations for its predictions, such as pointing out the ST-segment elevation in leads V1-V4, which is one of the hallmarks of AMI.
</p>

For more details, please refer to the [technical report](Final_Report.pdf).


## Abstract

Cardiovascular disease (CVD) is a primary cause of mortality globally, and the electrocardiogram (ECG) is a commonly used diagnostic tool for its detection. While Artificial Intelligence (AI) has shown an exceptional predictive ability for CVD, the lack of interpretability has deterred medical professionals from its use. To address this, we developed an explainable AI (XAI) framework that integrates ECG rules expressed in the form of first-order logic (FOL). The framework can uncover the underlying model's impressions of interpretable ECG features, which can be crucial for cardiologists to understand the diagnosis predictions generated by our system. Our experiments demonstrate the benefits of incorporating ECG rules into ECG AI such as improved performance and the ability to generate a diagnosis report that provides insights into how the model derived the predicted diagnoses. Overall, our XAI framework represents a great step forward in integrating domain knowledge into ECG AI models and enhancing their interpretability.



**Subject Descriptors**: Machine Learning, Neural-Symbolic Learning

**Keywords**: Explainable Artificial Intelligence, First-order Logic, Electrocardiogram

**Implementation Software**: Pytorch 2.0, Pytorch Lighting 1.9.4, NeuroKit2 0.2.3, Optuna 3.2
