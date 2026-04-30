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

### Todo

- [ ] Add a log file example for left-progressive sieving.
- [ ] Add an example for an LWE instance.

