# Periodic Complex Stochastic Processes for Retrieving Atomic Structures of Unknown Matters

Retrieving unknown atomic structures from observable analytical spectra or images remains a long-standing challenge across natural sciences. However, retrieval accuracy of existing retrieval methods on analytical data remains suboptimal because they have overlooked the underlying periodic quantum mechanical perturbations of non-equilibrium atomic structures behind analytical data. This paper proposes a periodic complex stochastic process (PCSP) that models such periodic perturbations and establishes theoretical backgrounds of periodic stochastic process in the complex-valued domain, including its sample diversity, process length, and periodicity. Finally, we develop a complex-valued cross-modal retrieval (CVCR) by integrating PCSP with cross-modal retrieval frameworks. CVCR outperformed existing cross-modal retrieval methods in cross-modal retrieval tasks of real-world analytical chemistry. Moreover, CVCR achieved state-of-the-art retrieval accuracy in zero-shot retrieval tasks on 40 million of molecules.


## Dataset
Due to the license issues, we uploaded a preprocessed .pt file of the NIST dataset instead of its raw data. You can download the preprocessed NIST dataset via https://drive.google.com/drive/folders/1Gi8XBxfyo2zfLhZyL1XE0IHwkXp1neZF.


## Run
Execute `exec.py` to train and evaluate CVCR with Normal-PCPS in a cross-modal retrieval task.
