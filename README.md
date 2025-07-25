# xpc-mammo
Simulation framework for propagation-based x-ray phase-contrast imaging of the breast &amp; material decomposition.


## 🧠 Overview

This guide will help you:

1. Clone this repository locally
2. Use [`uv`](https://github.com/astral-sh/uv) to manage a virtual environment
3. Install Python dependencies
4. Launch and use Jupyter notebooks

---

## 🔧 Setup Instructions

### 1. Clone the repository

First, open a terminal or command prompt and run:
```bash
git clone https://github.com/gjadick/xpc-mammo.git
cd xpc-mammo
```

If `git` is not installed:
- [Download and install Git](https://git-scm.com/downloads), then try again.

---

### 2. Install `uv` (if needed)

[`uv`](https://github.com/astral-sh/uv) is a fast, modern Python package manager that also handles virtual environments.

#### macOS or Linux:
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

#### Windows (PowerShell):
```bash
irm https://astral.sh/uv/install.ps1 | iex
```

After installation, restart your terminal and check it works:
```bash
uv --version
```

---

### 3. Create and activate a virtual environment

```bash
uv venv
source .venv/bin/activate  # macOS/Linux
```

Or on Windows:
```bash
.venv\Scripts\Activate
```

When activated, your terminal should show something like:
```bash
(.venv) your-computer:~/xpc-mammo $
```

---

### 4. Install dependencies
```bash
uv pip install -r requirements.txt
```

This installs all Python packages needed to run the code.

You will also need to install `chromatix` after this:
```bash
uv pip install git+https://github.com/chromatix-team/chromatix.git
```

If you want to use GPU acceleration, you'll want to specify which `jax` to install.
So if you have NVIDIA GPUs and CUDA 12 support, run `uv pip install -U "jax[cuda12]"`.

---

### 5. Launch the Jupyter Notebook interface

Start the notebook server:

    jupyter notebook

This will open a browser window where you can open and run `.ipynb` files in the project.

---

## 🧼 Deactivate the virtual environment

To deactivate the environment when you’re done, run in terminal:
```bash
deactivate
```
---

## ❓ Troubleshooting

If you run into problems:

- Make sure you’re running commands from inside the project directory (`xpc-mammo`)
- Make sure the virtual environment is activated before installing or running anything
- You can open an issue at: https://github.com/gjadick/xpc-mammo/issues

---

## ✅ Requirements

- Python 3.9 or later (`python --version`)
- Git
- [`uv`](https://github.com/astral-sh/uv) (for dependency and venv management)

If you prefer using `conda` or another package manager, feel free to adapt the steps.


