# BGJ-Sieve-GPU
An implementation of the **Becker–Gama–Joux** sieving algorithm, optimized for performance with **multi-GPU acceleration and external storage**. It demonstrates modern lattice sieving techniques described in:

> Ziyu Zhao, Jintai Ding, Towards Large-Scale Lattice Attack: New Lattice Records by Disk-Based Sieving

### When to Use
The main use case of `BGJ-Sieve-GPU` is attacks on small lattice-cryptography instances, but it should also perform well in any workload that requires heavy high-dimensional lattice reduction. Key features include:

- nearly all of the time-consuming steps required in practical lattice attacks are implemented
- AFAIK the lowest RAM usage and computational overhead among public lattice attack implementations as of 2026
- substantially better crash recovery

That said, it is not always the right choice:

- You need a CUDA-capable GPU to run `BGJ-Sieve-GPU`. 

  > Otherwise, I recommend [`BGJ-Sieve-AMX`](https://github.com/zhaoziyu0008/BGJ-Sieve-AMX), a CPU-only implementation with `AVX512` and optional `AMX` acceleration. Its memory usage and speed are similar to other GPU implementations except for this repository.

- Back up your data before running `BGJ-Sieve-GPU`. **You have been warned :)**

  > This project performs large amounts of sustained disk I/O, which can, in the worst case, cause irreversible data loss. If you are unsure whether you really need the performance of `BGJ-Sieve-GPU`, I still recommend `BGJ-Sieve-AMX`. If you have spare high-end GPUs or prefer a Python-facing workflow, [`G6K-GPU-Tensor`](https://github.com/WvanWoerden/G6K-GPU-Tensor) may also be a good option.

- If the input lattice contains large integers, I recommend using `fplll` for LLL preprocessing.

### Build
To build this repository, you need working `nvcc` and `g++` installations. The compiler must also be able to find the headers and libraries for `ntl`, `gmp`, and `libnuma`.

To build the binary, run:
```bash
make
```

Compilation may take a few minutes. After it finishes, the `hd_sieve` binary will be generated in `app/`.

### Usage

`hd_sieve` is the main executable for sieving-related workflows. It uses the same input and output lattice-basis format as `BGJ-Sieve-AMX`. Run it without arguments to see the help message:

```
$ cd app && ./hd_sieve
Cmd: ./hd_sieve 
Usage: ./hd_sieve --input INPUT_FILE --task TASK [OPTIONS]
Tasks:
  sieve:
    --TSD           Set target sieving dimension
    --CSD           Set current sieving dimension
    --MLD           Set min lifting dimension

  pump:
    --ind_l         Set left index of sieving context
    --ind_r         Set right index of sieving context
    --MSD           Set max sieving dimension
    --SSD           Set start sieving dimension
    --DH            Set dual hash ratio
    --DS            Set down sieve
    --output        Set output file

  dh:
    --POS           Set target position

  bkz:
    --BSD           Set BKZ sieving dimension
    --JUMP          Set jump step
    --D4F           Set dimension for free
    --BDH           Set dual hash ratio
    --STI           Set start index
```

Before running `hd_sieve`, you need to prepare the directories used to store the sieving database. The default paths are `.pool/0`, `.pool/1`, `.bucket/0`, `.bucket/1`, `.sol/0`, `.sol/1`, `.uid/0`, and `.uid/1`, corresponding to the figure in Section 5.2 of the paper. 

> These directories are not created automatically, to avoid accidentally overwriting existing data. If a run is interrupted for any reason, intermediate sieve data remains in these directories and can be reloaded through `Pool_hd_t::load()`. At the moment, when `--task` is set to `sieve` or `dh`, `hd_sieve` automatically detects existing data and attempts recovery. 

A minimal example can be:

```
$ cd tmp
$ mkdir -p .pool/0 .pool/1 .bucket/0 .bucket/1 .sol/0 .sol/1 .uid/0 .uid/1
$ ./hd_sieve --input INPUT_FILE --task sieve --TSD 110
[pool|26-04-29 13:48:13.955] enter keyfunc sampling
[pool|26-04-29 13:48:14.080] exit keyfunc sampling
[pool|26-04-29 13:48:14.080] enter keyfunc check
...
[pool|26-04-29 13:49:43.312] exit keyfunc bgj1_Sieve_hd
```

Before running any serious tasks, generally you should adjust `config.h` to match your hardware and get the best performance. You can control both the computational and storage resources used by `hd_sieve`. Compile again after modifying `config.h` to apply the changes.

### Example: Reproducing the LWE Challenge

This section gives a step-by-step example for reproducing the solution of one LWE challenge from the paper. Many of the concrete parameters here are not strict: adjusting them slightly will usually not make the overall computation dramatically faster or slower. The full process is expected to take a few days, depending on your machine configuration.

1. Download a challenge instance.

```bash
$ wget https://www.latticechallenge.org/lwe_challenge/challenges/LWE_45_035.txt
```

2. Use its first 213 samples to construct a unique-SVP instance named `L_4535`.

3. Apply a light BKZ reduction with `BGJ_SIEVE_AMX` to obtain something like `L_tmp0`.

4. Run several rounds of heavy BKZ reduction with `hd_sieve`:

```bash
$ ./hd_sieve --input L_tmp0 --task bkz --BSD 115 --JUMP 8 --D4F 30 --STI 10 --output L_tmp1
$ ./hd_sieve --input L_tmp1 --task bkz --BSD 122 --JUMP 8 --D4F 32 --STI 5 --output L_tmp2
$ ./hd_sieve --input L_tmp2 --task bkz --BSD 129 --JUMP 8 --D4F 35 --STI 0 --output L_tmp3
```

Note: the files `L_tmp1`, `L_tmp2`, and `L_tmp3` are also included in the `example/` directory. Since sieving is randomized, your intermediate results may look very different from mine, but the basis quality should be comparable.

5. Run one large sieving job:

```bash
$ ./hd_sieve --input L_tmp3 --task sieve --MLD 135
```

Once the sieving dimension exceeds 135, this command prints the current shortest-vector length at each dimension. Around sieving dimension 153, you are expected to find something like:

```text
[62 67 9 36 -47 37 -49 12 -86 58 52 -58 -50 -7 -4 53 -13 61 43 96 89 6 41 46 111 48 41 23 84 7 71 -65 64 -33 64 87 11 -113 31 0 46 6 -34 95 -52 -36 -26 11 -150 -93 -78 -118 53 -35 -16 -75 -13 59 -70 197 -83 102 44 69 20 137 21 -93 -20 49 46 73 -83 -73 -151 115 67 -28 18 -19 17 7 -110 38 -92 -24 -3 -26 -108 17 -24 -5 6 48 20 -91 -84 32 -25 -18 40 39 98 13 36 24 44 40 -184 131 48 -7 -37 112 137 30 -115 7 42 -10 -14 -50 51 -25 -16 -5 -72 -131 49 -39 -79 -16 -8 63 4 -3 -21 81 3 59 55 23 4 -12 -10 7 -74 35 133 -86 -155 -66 -82 14 4 2 55 80 -15 -20 137 -186 106 202 21 14 -48 -82 -25 150 3 153 -40 70 -9 -15 -2 71 42 36 30 132 -28 4 184 -102 -30 -12 -71 40 -23 2 -115 23 -30 12 -9 -79 -13 -43 -57 -20 35 36 74 102 81 33 -57 -92 -13 9 -70.945]
```

After obtaining the short vector, you can easily recover the secret:

```text
[1635 312 1536 1180 158 319 1028 117 1113 760 649 12 1383 411 35 323 1399 97 168 1666 796 724 1451 1670 1830 1031 1863 165 52 614 146 813 1492 616 119 1331 1595 1803 1226 1095 1326 847 963 1085 2016]
```

