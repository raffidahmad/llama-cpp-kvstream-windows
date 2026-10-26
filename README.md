# llama.cpp ROCm KV-streaming fork: Windows build

Windows build recipe for [achyudh/llama.cpp](https://github.com/achyudh/llama.cpp), branch `feature/adaptive-kv-stream-rocm`
(`--kv-stream`: keeps most of the KV cache in system RAM and streams it to the GPU, so a 16 GB RDNA 4 card can run
27B models at 262k context). The upstream README only covers Linux.

This repo contains **no llama.cpp source**. It holds a small patch (`patches/`), a build script and prebuilt binaries
(see [Releases](../../releases)). All credit for the fork goes to achyudh; the fork builds on
[RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)
and [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts).

## What the patch changes

`patches/0001-windows-hip-no-hsa.patch` (13 lines). The fork's copy engine uses the Linux-only HSA runtime
(`hsa-runtime64`). On Windows the patch skips that dependency and compiles the HSA code path out
(`ggml/src/ggml-hip/CMakeLists.txt`, `ggml/src/ggml-cuda/copy-engine.cu`), and maps `cudaEventQuery` to `hipEventQuery`.
Nothing else is modified.

## Requirements

- Windows 11, AMD Adrenalin driver with ROCm/HIP support for your GPU
- RDNA 4 (RX 9070 / 9070 XT = `gfx1201`) is what this was built and tested on. Other GPUs: pass `-Gpu <target>` to `build.ps1` (untested)
- [Git](https://git-scm.com/), [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for the venv)
- Visual Studio 2022 Build Tools with **Desktop development with C++** (provides CMake, Ninja, Windows SDK, MSVC libs for the linker)
- ~10 GB free disk (ROCm SDK wheels expand to several GB), 32 GB RAM recommended for 262k context

## The ROCm venv

There is no system-wide ROCm install on Windows. The compiler and runtime come from AMD's pip wheels (ROCm 10.1.0,
the "whl-next" index) in a local venv. `build.ps1` does this for you; manually:

```powershell
uv venv --python 3.12 .venv-rocm
uv pip install --python .venv-rocm\Scripts\python.exe --index-url https://stable.repo.amd.com/rocm/whl-next/ `
    rocm==10.1.0 rocm-sdk-core==10.1.0 rocm-sdk-devel==10.1.0 rocm-sdk-libraries==10.1.0 rocm-sdk-device-gfx1201==10.1.0
.venv-rocm\Scripts\rocm-sdk.exe init      # expands the devel package (~1.3 GB)
```

At run time the server needs `_rocm_sdk_core\bin`, `_rocm_sdk_devel\bin` and `_rocm_sdk_libraries\bin` from that venv on
`PATH`, plus `HIP_PATH` (see `run-example.ps1`). If you only run the prebuilt binary you can skip `rocm-sdk-devel` and the compiler.

## Prebuilt release (no setup)

Download the zip from [Releases](../../releases), extract it, and run:

```powershell
powershell -ExecutionPolicy Bypass -File run-example.ps1 -Model C:\models\your-model.gguf -Device ROCm0
```

It contains `llama-server.exe`, `llama-bench.exe` and every ROCm runtime library they need (HIP, comgr, hiprtc, rocBLAS,
hipBLASLt, hipBLAS, rocSOLVER and the `gfx1201` kernel files). You only need the AMD Adrenalin driver and the
[Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe). It is built for `gfx1201` (RX 9070 / 9070 XT) only;
other GPUs need to build from source.

## Build

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
```

This creates the venv (if missing), clones achyudh/llama.cpp into `src\` at the pinned commit, applies the patch,
builds `llama-server` and `llama-bench` into `src\build-rocm\bin`, and copies the SDK's HIP DLLs next to the exes.
Only the pinned commit is fetched (no git history). To make the release zip, run `package.ps1` afterwards; it writes `dist\*.zip`
and fails if any DLL import is unresolved.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File run-example.ps1 -Model C:\models\your-model.gguf -Ctx 262144
```

Check the startup log for device names. With an iGPU present the dGPU may be `ROCm1` (`-Device ROCm1`).
The script enables MTP + n-gram speculative decoding by default (the settings used in the Results below). That needs a model with
an MTP head; pass `-NoSpec` for other models.
Flags beyond the example (MTP, n-gram speculation, reasoning) are documented in
[achyudh's README](https://github.com/achyudh/llama.cpp/tree/feature/adaptive-kv-stream-rocm).
The HIP queue variables set by the script (GPU_MAX_HW_QUEUES=2, DEBUG_HIP_FORCE_ASYNC_QUEUE=1, DEBUG_HIP_FORCE_GRAPH_QUEUES=1) gave a small but consistent gain in an A/B test (see Results).

## Updating from upstream

1. Pick the new commit on `achyudh/llama.cpp` and set `-Commit` (or the default in `build.ps1`).
2. Run `build.ps1`. If the patch no longer applies, redo the two small edits and regenerate it with `git diff > patches\0001-windows-hip-no-hsa.patch`.
3. Check the fork's README for new flags.

## Results

Measured on: RX 9070 16 GB (`gfx1201`, the dGPU is `ROCm1` because an iGPU is present), Ryzen 9 7900, 32 GB RAM, Windows 11.
These are **single runs** on short prompts (about 9k and 21k tokens), greedy sampling, 512 generated tokens for the
code prompt and 256 for the 9k/21k prompts. They are **not** filled 64k to 262k contexts, and output quality was not
evaluated. Run-to-run noise is about ±2-3 t/s. Each cell is decode t/s (TG) or prefill t/s (PP).

Common flags: `--device ROCm1 -fit off --flash-attn on`, K q8_0 / V q4_0 (`--kv-stream` only accepts this pair),
batch 2048 / ubatch 512, `--spec-type draft-mtp,ngram-mod --spec-rs-replay`, MTP n=4, n-gram 12/32/24, probabilistic
draft sampling, HIP env vars from `run-example.ps1`.

### Best config per model (this fork, Windows)

| Model | Context | `--kv-resident-mib` (prefill / decode) | Code TG | 9k TG | 21k TG | 21k PP |
|---|---|---|---|---|---|---|
| Qwen3.8-27B ASCII-Condensed UD-IQ4_XS (12.63 GiB) | 262144 | 300 / 1800 | **81.5** | 75.1 | 94.6 | ~1009 |
| Qwen3.8-27B GSQ-RCO IQ3_S (11.29 GiB) | 262144 | 300 / 1800 | 65.2 | 61.9 | 77.7 | ~966 |
| Qwen3.8-27B GSQ-RCO IQ3_S | 131072 | 450 / 2420 | 67.9 | 64.6 | 77.1 | ~1038 |

### This fork vs. Vulkan llama.cpp b11873 (GSQ-RCO IQ3_S)

Vulkan: MTP n=2, ubatch 256, K/V q8_0. ROCm rows: K q8_0 / V q4_0, ubatch 512.

| Backend | Context | Code TG | 9k PP / TG | 21k PP / TG |
|---|---|---|---|---|
| Vulkan | 131072 | 46.0 | 594 / 42.9 | 588 / 41.0 |
| Vulkan | 262144 | 30.8 | 325 / 29.3 | 302 / 29.5 |
| ROCm fork, no `--kv-stream` (MTP n=2) | 131072 | 52.6 | 413 / 35.7 | 353 / 24.0 |
| ROCm fork, `--kv-stream` (MTP n=2) | 131072 | 61.7 | 1086 / 59.1 | 1054 / 54.6 |
| ROCm fork, `--kv-stream` (MTP n=2) | 262144 | 23.7 | 963 / 30.2 | 999 / 28.2 |
| ROCm fork, `--kv-stream`, MTP n=4 + n-gram, resident 300/1800 | 262144 | 65.2 | 999 / 61.9 | 966 / 77.7 |

Speculative settings were the biggest lever at 262k (code decode 21 to 65-71 t/s). The n-gram drafter favours prompts
whose output repeats the context, which is why the 21k summarisation prompt decodes faster than the code prompt.

### HIP environment variables (ASCII IQ4_XS, 262144 ctx, resident 300/1800; two runs each)

`GPU_MAX_HW_QUEUES=2`, `DEBUG_HIP_FORCE_ASYNC_QUEUE=1`, `DEBUG_HIP_FORCE_GRAPH_QUEUES=1`

| HIP env vars | Code TG | 9k PP / TG | 21k PP / TG |
|---|---|---|---|
| set, run 1 | 83.5 | 1042 / 78.2 | 1076 / 89.1 |
| set, run 2 | 81.9 | 1084 / 73.8 | 1086 / 87.7 |
| not set, run 1 | 80.2 | 1068 / 73.4 | 1037 / 84.2 |
| not set, run 2 | 80.2 | 1070 / 73.7 | 1038 / 84.2 |

About +2-4% code decode and +3-5% at 21k; prefill is equal. Small, but consistent across these four runs.

### Notes and caveats

- **Crashes:** the IQ4_XS model crashed (`ROCm error: unspecified launch failure`) with resident 450/2420; use 300/1800.
- **VRAM:** with resident 450/2420 the 262k run on the IQ3_S model showed 15.7 GB used, probably spilling into shared memory. 300/1800 brought it down to 14.2 GB.
- **Windows overhead:** Windows keeps some VRAM for the desktop. Moving the display to an iGPU was not tested.
- **No SDMA overlap:** the HSA copy engine does not exist on Windows, so the fork's SDMA transfer overlap is not in effect here; the portable `hipMemcpyAsync` path is used instead.
- **DLLs:** `amdhip64_7.dll`, `amd_comgr.dll` and `hiprtc*.dll` from the SDK must sit next to the exe. Otherwise Windows loads the older driver copy from System32 and the server fails with "device kernel image is invalid". `build.ps1` copies them.

## Contribute test results

This has only been tested on one machine (RX 9070, see Results). **Reports from other cards are very welcome, including failures.**
Especially wanted: RX 9070 XT, RX 9060 XT, RX 7000 series, RX 6000 series (for example the RX 6700 XT), and machines with less RAM.
Whether ROCm for Windows supports your card at all is an open question, so "does not build" is a useful result too.

1. **Get the binary.** Use the [release zip](../../releases) if you have an RX 9070 / 9070 XT. For any other GPU, build from source with your target, for example the RX 6700 XT is `gfx1031`:
   `powershell -ExecutionPolicy Bypass -File build.ps1 -Gpu gfx1031` (the AMD wheel index may not have kernels for every target).
2. **Start the server** with a model that has an MTP head (or add `-NoSpec`):
   `powershell -ExecutionPolicy Bypass -File run-example.ps1 -Model C:\models\model.gguf -Device ROCm0 -Port 8080`
   Check the startup log for the device name; with an iGPU the dGPU is usually `ROCm1`. If you run out of VRAM, lower `-ResidentMiB` / `-ResidentDecodeMiB` or `-Ctx`.
3. **Run the benchmark** in a second window (about a minute):
   `powershell -ExecutionPolicy Bypass -File bench.ps1 -Port 8080`
   It prints a markdown block with your GPU, driver, RAM and results.
4. **Open an issue** with the [Test results form](../../issues/new?template=test-results.yml) and paste the block.
   Questions and "will it work on my card?" go to [Discussions](../../discussions).

Notes for comparable numbers: close other GPU-heavy programs, use the same flags as `run-example.ps1` unless you say what you changed,
and expect prompt-to-prompt variation of a few tokens per second (the three code prompts show how much the speculative drafts help each one).

## License

MIT, same as llama.cpp (see `LICENSE`).
