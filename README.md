
# **cosmos – MCMC for Cosmology**

`cosmos` is a complete toolkit for performing **late-time cosmological analysis** using **Markov Chain Monte Carlo (MCMC)** sampling.
It provides a flexible framework for both analytical and numerical cosmological models, along with ready-to-use likelihoods and example scripts.

---

## Features

* **MCMC Sampler** – robust and efficient sampling for cosmological parameter estimation.
* **Chebyshev Method** – fast solver for algebraic models.
* **RK4 Solver** – fourth-order Runge–Kutta integration for differential models.
* **Built-in Likelihoods** for:

  * Cosmic Chronometers (**CC**) (Moresco et al. 2020, https://arxiv.org/abs/2003.07362, https://gitlab.com/mmoresco/CCcovariance)
  * Type Ia Supernovae (**SN1a**) (Riess et al. 2021, https://arxiv.org/abs/2112.03863, https://pantheonplussh0es.github.io)
  * Baryon Acoustic Oscillations (**BAO**) (DESI Collaboration 2024, https://arxiv.org/pdf/2404.03002)

---

## Requirements

* **Eigen3** (libeigen3-dev)
* **OpenMP** (libomp-dev)
* **MPI** (libopenmpi-dev)
* **BLAS** (libopenblas-dev)
* **lapacke** (liblapacke-dev)

---

## Examples

You’ll find three fully worked examples in the `/examples` directory:

| Example                    | Type         | Description                                |
| -------------------------- | ------------ | ------------------------------------------ |
| `LCDM_example`             | Analytic     | Standard ΛCDM cosmology                    |
| `PowerLaw_example`         | Algebraic    | Power-law ( f(T) ) gravity model           |
| `quintessence_example`     | Differential | Exponential quintessence dark energy model |

There’s also a **Python analysis and plotting script** (requires [`chainconsumer`](https://samreay.github.io/ChainConsumer)) demonstrating how to visualize and analyze your MCMC chains.

---

## Adding Your Own Models

To add your own models, create a new folder under `/user` and add a `CMakeLists.txt` file there.
You can register your model’s executable using the provided CMake macro:

```cmake
# Add a custom MCMC executable
add_cosmos_executable(your_executable_name
    your_executable_file.cxx   # (important: path must be relative to /user/)
)
```

This automatically links your model to the core `cosmos` framework and external dependencies.

---

## Parallel Execution

`cosmos` supports both **multithreading** (via OpenMP) and **multiprocessing** (via OpenMPI) to take advantage of modern multi-core systems.

### OpenMP (Multithreading)

You can control the number of threads by setting the environment variable `OMP_NUM_THREADS` before running the executable:

```bash
OMP_NUM_THREADS=6 LCDM_example
```

### OpenMPI (Multiprocessing)

Run the executable using `mpirun` to launch multiple processes:

```bash
mpirun -np 4 LCDM_example --map-by core
```

Use whichever mode best suits your hardware and workload.

---

## Further Reading

For detailed discussions on the **mathematical foundations of MCMC**, the **specific algorithms** implemented here, and **performance benchmarks**,
please refer to **`report.pdf`** (once available).

---

## Notes

* For the **BAO likelihood**, an example implementation of the required `GetRDrag` function can be found in the `LCDM_example` script.
  In practice it was found to only work well for ΛCDM or models with a close ΛCDM limit though.

---

## Happy Experimenting!

We hope `cosmos` helps you explore, test, and refine your cosmological models efficiently and enjoyably.

