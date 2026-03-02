# Verified Z-Image 

Develop a verified version of z-image diffusion network

## Motivation
- Assume the GPU is not secured, which might be attached or not really doing the computation by giving random results. 
- GPU is connected with a large TEE CPU
- Only CPU is trusted, we only use GPU to do matrix acceleration.

## Support functions
- Verify all matmul/linear operations results computed by GPU by moving the results to CPU and verify with freivalds algorithm.
- Key processing
    - Run the whole model on CPU, and also copy to GPU
    - Start Run model on CPU and GPU at the same time
    - On CPU: matmul/linear only do verify, other operations use CPU to compute
    - On GPU: matmul/linear computation as normal, but send the output data of matmul/linear to CPU for verification.
    - Asynch execute the transfer data from GPU to CPU and next GPU computations.

- Apply to all modules include matmul or linear, both Attention and FFN layers.


## Profile
- Include profiling tool for getting performance
- Switch profiling on/off

## Test
- Write test to verify accuracy, compare the results with the original modules or pipeline.
- Different level of test
    - Module level
    - Transformer level
    - Pipeline Level


# Verified Z-Image

Develop a **verified version of the Z-Image diffusion network** with GPU-accelerated matrix operations while ensuring correctness via CPU verification.

---

## Motivation

* **Untrusted GPU:** Assume the GPU is not trusted—it may be physically attached but could produce incorrect or random results.
* **Trusted CPU:** Only the CPU is trusted; a large Trusted Execution Environment (TEE) CPU is available.
* **Goal:** Leverage GPU for acceleration of matrix-heavy operations while guaranteeing correctness by verification on CPU.

---

## Architecture Overview

1. **Dual Execution:**

   * CPU: trusted, performs verification for GPU outputs and handles operations not offloaded to GPU.
   * GPU: untrusted, performs accelerated matmul/linear computations.

2. **Operations Offloaded for Verification:**

   * Only **matmul/linear operations** are offloaded to GPU.
   * Other operations (activations, normalization, element-wise functions) are executed directly on CPU.

3. **Verification Strategy:**

   * GPU outputs for matmul/linear operations are **asynchronously transferred** to CPU.
   * CPU verifies correctness using **Freivalds’ algorithm** (probabilistic verification for matrix multiplication).
   * CPU can recompute full operation if verification fails.

4. **Pipeline Integration:**

   * Apply verification to **all modules**: Attention (query/key/value projections, output linear), Feed-Forward Network (FFN) layers, and any other linear/matmul operation in the network.

---

## Workflow

1. **Model Initialization:**

   * Load model on CPU.
   * Copy model weights to GPU for acceleration.

2. **Execution Loop:**
   For each layer in the model:

   **On GPU:**

   * Run as normal
   * Perform matmul/linear operation as normal.
   * Send the result asynchronously to CPU for verification.

   **On CPU:**

   * For matmul/linear: receive GPU output and verify using Freivalds’ algorithm.
   * For other operations: compute as normal.

   **Synchronization:**

   * Ensure verification of a previous operation does not block GPU computations of the next operations (overlap computation and verification).

---

## Key Processing Details

* **Verification of Matmul/Linear:**

  1. Randomly select a small number of vectors for Freivalds’ algorithm (probabilistic).
  2. Compute `C_CPU * v` and compare with `C_GPU * v`.
  3. If verification passes, continue; if fails, recompute fully on CPU.

* **Asynchronous Data Transfer:**

  * Use non-blocking GPU → CPU transfer for verified outputs.
  * Ensure GPU pipeline remains busy while CPU verifies.

* **Layer Coverage:**

  * Attention layers: Q/K/V projections, attention output linear.
  * FFN layers: hidden projection, output projection.
  * Any custom matmul/linear operation in the model.

---

## Profiling

* Include a **profiling module** to measure performance:

  * GPU computation time per matmul/linear operation.
  * CPU verification time.
  * Asynchronous transfer overhead.
  * Total end-to-end latency.

* **Profiling Control:**

  * Allow switching profiling on/off via configuration or runtime flag.
  * Output results in CSV or JSON for analysis.

---

## Testing

* **Correctness Verification:**

  * Compare verified model outputs with **baseline unmodified model** on CPU.

* **Test Levels:**

  1. **Module Level:** test individual linear/matmul operations.
  2. **Transformer Level:** test attention + FFN blocks.
  3. **Pipeline Level:** test full diffusion model end-to-end.

* **Test Scenarios:**

  * Correct GPU outputs.
  * Simulated corrupted GPU outputs (to ensure verification detects failures).
  * Performance comparison (with and without verification).

* **Metrics:**

  * Accuracy deviation.
  * Detection rate of incorrect GPU outputs.
  * Latency overhead due to verification.

---

## Optional Enhancements

* **Adaptive Verification:**

  * Dynamically adjust frequency of verification based on GPU reliability (e.g., only verify a subset of matmuls in each batch).

* **Hybrid Computation:**

  * Some sensitive layers can be fully computed on CPU, others offloaded to GPU with verification.

* **Fault Logging:**

  * Record any verification failures for later analysis.
