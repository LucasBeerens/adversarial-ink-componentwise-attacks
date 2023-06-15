# Adversarial Ink: Componentwise Attacks

This repository contains the code and experiments conducted for the paper titled ["Adversarial Ink: Componentwise Backward Error Attacks on Deep Learning"](https://doi.org/10.48550/arXiv.2306.02918) to appear in IMA Journal of Applied Mathematics. In this paper, we address the vulnerability of deep neural networks to adversarial attacks, specifically focusing on handwritten documents or printed texts where ink consistency alterations can change the classification results while appearing natural to the naked eye. The algorithms described in the paper can be found in this repository.

## Abstract

Deep neural networks are capable of state-of-the-art performance in many classification tasks. However,
they are known to be vulnerable to adversarial attacks—small perturbations to the input that lead
to a change in classification. We address this issue from the perspective of backwarkd error and
condition number, concepts that have proved useful in numerical analysis. To do this, we build on
the work of Beuzeville et al. (2021). In particular, we develop a new class of attack algorithms that
use componentwise relative perturbations. Such attacks are highly relevant in the case of handwritten
documents or printed texts where, for example, the classification of signatures, postcodes, dates or
numerical quantities may be altered by changing only the ink consistency and not the background. This
makes the perturbed images look natural to the naked eye. Such “adversarial ink” attacks therefore reveal
a weakness that can have a serious impact on safety and security. We illustrate the new attacks on real
data and contrast them with existing algorithms. We also study the use of a componentwise condition
number to quantify vulnerability.

## Repository Structure

The repository is organized as follows:

- **`data/`**: Will the datasets used in the experiments.
- **`src/`**: The source code for the proposed adversarial ink attacks and vulnerability analysis. The directory structure is as follows:
  - **`data_attacks/`**: Implementations of the attack algorithms.
  - **`load_data/`**: Code used to load the different datasets.
  - **`models/`**: The different neural networks that are used in experiments.
  - **`utils/`**: Utility functions and helpers used throughout the experiments.
- **`notebooks/`**: Jupyter notebooks or scripts showcasing the experiments conducted in the paper. Detailed instructions and explanations are provided within.
- **`resultsNumpy/`**: Stores the results obtained from running the experiments in NumPy form.
- **`resultsFinal/`**: Stores resulting images from experiment analysis.

## Getting Started

To reproduce the experiments conducted in the paper, follow these steps:

1. Clone the repository to your local machine:

   ```shell
   git clone https://github.com/LucasBeerens/adversarial-ink-componentwise-attacks.git
   cd adversarial-ink-componentwise-attacks
   ```

2. Install the necessary dependencies. We recommend using a virtual environment:

   ```shell
   python3 -m venv env
   source env/bin/activate  # (for Unix/Linux)
   # OR
   env\Scripts\activate  # (for Windows)
   ```

   Then, install the dependencies using `pip`:

   ```shell
   pip install -r requirements.txt
   ```


## Citation

If you use this code or find our research useful, please consider citing:

```
@article{beerens2023adversarial,
      title={Adversarial Ink: {C}omponentwise Backward Error Attacks on Deep Learning}, 
      author={Lucas Beerens and Desmond J. Higham},
      year={2023},
      journal={IMA Journal of Applied Mathematics},
      volume={to appear}
}
```