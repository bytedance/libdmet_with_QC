Quantum computing unitary coupled-cluster singles and doubles (QC-UCCSD) solvers with two different backends, FQE and Yao, are provided in this package.
* For the usage of UCCSD solver with FQE:
  1. Directly install this package via the supplied setup.py file.
  `pip3 install -e .`
  2. QC-UCCSD solver with FQE is directly available.
  ```
  from libdmet_qc.solver.fqe_qc import QCfqe
  ```

* For the usage of UCCSD solver with Yao, a bit more effort is needed after correctly installation of this package:
  1. Download and install [Julia](https://julialang.org/downloads/)(1.7 or higher) on your system.
  2. Add julia into your PATH, for example,
  `export PATH=your_julia_path/julia-1.8.0-rc1/bin:$PATH`
  3. Launch julia REPL and enter the Pkg REPL by pressing `]`.
  ```
  julia
  ]
  ```
  4. Install the following depedencies.
  ```
  add Distributed BenchmarkTools@1.3.1 PyCall@1.93.1 TimerOutputs@0.5.19 Yao@0.7.4 YaoExtensions@0.2.5
  ```
  5. Change the `solver_path` in `path/libdmet/solver/yao_qc.py` to the path of your `CPU_QCsolver_UCCSD.jl`.
  6. QC-UCCSD solver with Yao is available then.
  ```
  from libdmet_qc.solver.yao_qc import QCyao
  ```
