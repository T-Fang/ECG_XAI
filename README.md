# ECG_XAI

This repository contains the implementation for my Final Year Project at NUS. In this project, I developed a scalable explainable AI (XAI) framework for electrocardiogram (ECG) diagnosis. The framework can easily incorporate ECG rules expressible in First-order Logic (FOL), automatically capture intermediate interpretable ECG features, and finally produce a diagnosis report comprehensible to cardiologists. The detailed report can be found [here](Final_Report.pdf).



## Abstract

Cardiovascular disease (CVD) is a primary cause of mortality globally, and the electrocardiogram (ECG) is a commonly used diagnostic tool for its detection. While Artificial Intelligence (AI) has shown an exceptional predictive ability for CVD, the lack of interpretability has deterred medical professionals from its use. To address this, we developed an explainable AI (XAI) framework that integrates ECG rules expressed in the form of first-order logic (FOL). The framework can uncover the underlying model's impressions of interpretable ECG features, which can be crucial for cardiologists to understand the diagnosis predictions generated by our system. Our experiments demonstrate the benefits of incorporating ECG rules into ECG AI such as improved performance and the ability to generate a diagnosis report that provides insights into how the model derived the predicted diagnoses. Overall, our XAI framework represents a great step forward in integrating domain knowledge into ECG AI models and enhancing their interpretability.



**Subject Descriptors**: Machine Learning, Neural-Symbolic Learning

**Keywords**: Explainable Artificial Intelligence, First-order Logic, Electrocardiogram

**Implementation Software**: Pytorch 2.0, Pytorch Lighting 1.9.4, NeuroKit2 0.2.3, Optuna 3.2
