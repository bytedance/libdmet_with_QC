# libDMET_with_QC
libDMET (pyscf 2.1) with unitary CCSD solvers of quantum computing chemistry.

## Installation
1. Please refer to [libdmet_preview](https://github.com/gkclab/libdmet_preview) for the installation of libDMET. 
2. Please refer to [QCSOLVERS.md](./QCSOLVERS.md), for the configuration of quantum computing chemistry solvers.
#### Prerequisites
- libdmet
- pyscf 2.1 or higher
- matplotlib 3.7.1 or higher
- julia 0.5.7 or higher
- numpy 1.21
- openfermion 1.3.0
- openfermionpyscf 0.5
- fqe 0.2.0
- projectq 0.7.3 


## Reference
The following papers should be cited in publications utilizing the libDMET program package:  
- Cui, Z.-H., Zhu, T., Chan, G. K.-L. Efficient Implementation of Ab Initio Quantum Embedding in Periodic Systems: Density Matrix Embedding Theory. [J. Chem. Theory Comput. 16, 119–129 (2020)](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00933).  
- Zhu, T., Cui, Z.-H., Chan, G. K.-L. Efficient Formulation of Ab Initio Quantum Embedding in Periodic Systems: Dynamical Mean-Field Theory. [J. Chem. Theory Comput. 16, 141–153 (2020)](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00934).

If you use QC solvers in your research, please cite this paper:
- Changsu Cao, Jinzhao Sun, Xiao Yuan, Han-Shi Hu, Hung Q Pham, Dingshun Lv, Ab initio Quantum Simulation of Strongly Correlated Materials with Quantum Embedding. [arXiv:2209.03202v2 (2023)](https://doi.org/10.48550/arXiv.2209.03202).  

Specifically, if you use the quantum computing solver based on FQE (QCfqe), please also cite:  
- Rubin, N. C. et al. The Fermionic Quantum Emulator. [Quantum 5, 568 (2021)](https://quantum-journal.org/papers/q-2021-10-27-568/).  

If you use the quantum computing solver based on Yao (QCyao), please also cite:  
- Luo, X.-Z., Liu, J.-G., Zhang, P, Wang, L. Yao.jl: Extensible, Efficient Framework for Quantum Algorithm Design. [Quantum 4, 341 (2020)](https://quantum-journal.org/papers/q-2020-10-11-341/).
