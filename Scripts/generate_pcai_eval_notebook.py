import json
import datetime
from pathlib import Path

project_root = Path(r"C:\Users\david\PC_AI")
nb_dir = project_root / "Notebooks"
nb_dir.mkdir(exist_ok=True)

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
nb_path = nb_dir / f"pcai_eval_{now}.ipynb"

cells = []
cell_counter = 1


def md_cell(lines):
    global cell_counter
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "id": f"md-{cell_counter}",
        "source": lines,
    }
    cell_counter += 1
    return cell


def code_cell(lines):
    global cell_counter
    cell = {
        "cell_type": "code",
        "metadata": {},
        "id": f"code-{cell_counter}",
        "source": lines,
        "outputs": [],
        "execution_count": None,
    }
    cell_counter += 1
    return cell


cells.append(md_cell([
    "# PC-AI Comprehensive LLM Evaluation Suite\n",
    "\n",
    "This notebook drives a parameter sweep across PC-AI compiled backends, captures system resource telemetry, and graphs performance/quality metrics. It also includes optional hooks for standard LLM benchmarks via `lm-evaluation-harness`.\n",
]))

cells.append(code_cell([
    "import os, json, time, datetime, subprocess, sys, socket\n",
    "from pathlib import Path\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import urllib.request\n",
    "\n",
    "PROJECT_ROOT = Path(r'C:\\\\Users\\\\david\\\\PC_AI')\n",
    "REPORTS_DIR = PROJECT_ROOT / 'Reports' / 'llm-eval' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')\n",
    "REPORTS_DIR.mkdir(parents=True, exist_ok=True)\n",
    "print('Reports:', REPORTS_DIR)\n",
    "\n",
    "def find_powershell():\n",
    "    for exe in ['pwsh', 'powershell']:\n",
    "        try:\n",
    "            subprocess.run([exe, '-NoProfile', '-Command', '$PSVersionTable.PSVersion'], check=True, capture_output=True, text=True)\n",
    "            return exe\n",
    "        except Exception:\n",
    "            continue\n",
    "    raise RuntimeError('PowerShell not found')\n",
    "\n",
    "POWERSHELL = find_powershell()\n",
    "print('Using PowerShell:', POWERSHELL)\n",
    "\n",
    "def run_ps(cmd, check=True):\n",
    "    result = subprocess.run([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', cmd],\n",
    "                            capture_output=True, text=True)\n",
    "    if check and result.returncode != 0:\n",
    "        raise RuntimeError(f'PowerShell failed: {result.stderr}\\nSTDOUT:\\n{result.stdout}')\n",
    "    return result\n",
    "\n",
    "def get_free_port():\n",
    "    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:\n",
    "        s.bind(('', 0))\n",
    "        return s.getsockname()[1]\n",
]))

cells.append(md_cell([
    "## Load PC-AI modules\n",
    "This ensures the evaluation and performance modules are available.\n",
]))

cells.append(code_cell([
    "ps = f'''\n",
    "Import-Module '{PROJECT_ROOT / 'Modules' / 'PcaiInference.psd1'}' -Force -ErrorAction SilentlyContinue\n",
    "Import-Module '{PROJECT_ROOT / 'Modules' / 'PC-AI.Evaluation' / 'PC-AI.Evaluation.psd1'}' -Force\n",
    "Import-Module '{PROJECT_ROOT / 'Modules' / 'PC-AI.Performance' / 'PC-AI.Performance.psd1'}' -Force\n",
    "Get-PcaiInferenceStatus | Format-List\n",
    "'''\n",
    "print(run_ps(ps).stdout)\n",
]))

cells.append(md_cell([
    "## Model + binary discovery\n",
    "We auto-detect a test model and compiled backend binaries. For llama.cpp we need a `.gguf`. For mistral.rs, a SafeTensors model is acceptable. Set `providers.pcai-native.modelPath` in Config/llm-config.json to override.\n",
]))

cells.append(code_cell([
    "def load_model_path_from_config():\n",
    "    config_path = PROJECT_ROOT / 'Config' / 'llm-config.json'\n",
    "    if config_path.exists():\n",
    "        try:\n",
    "            cfg = json.loads(config_path.read_text())\n",
    "            model_path = cfg.get('providers', {}).get('pcai-native', {}).get('modelPath')\n",
    "            if model_path and Path(model_path).exists():\n",
    "                return Path(model_path)\n",
    "        except Exception:\n",
    "            pass\n",
    "    return None\n",
    "\n",
    "def find_model_path():\n",
    "    config_path = load_model_path_from_config()\n",
    "    if config_path:\n",
    "        return config_path\n",
    "\n",
    "    models_dir = PROJECT_ROOT / 'Models'\n",
    "    gguf = list(models_dir.rglob('*.gguf'))\n",
    "    if gguf:\n",
    "        return gguf[0]\n",
    "\n",
    "    gemma = models_dir / 'functiongemma-270m-it' / 'model.safetensors'\n",
    "    if gemma.exists():\n",
    "        return gemma\n",
    "\n",
    "    lm = Path(os.environ.get('LOCALAPPDATA','')) / 'lm-studio' / 'models'\n",
    "    if lm.exists():\n",
    "        candidates = list(lm.rglob('*.gguf'))\n",
    "        if candidates:\n",
    "            return candidates[0]\n",
    "\n",
    "    ollama = Path.home() / '.ollama' / 'models' / 'blobs'\n",
    "    if ollama.exists():\n",
    "        candidates = list(ollama.rglob('*.gguf'))\n",
    "        if candidates:\n",
    "            return candidates[0]\n",
    "\n",
    "    return None\n",
    "\n",
    "def find_binary(name):\n",
    "    candidates = [\n",
    "        PROJECT_ROOT / 'bin',\n",
    "        Path.home() / '.local' / 'bin',\n",
    "        PROJECT_ROOT / 'Native' / 'pcai_core' / 'pcai_inference' / 'target-ffi' / 'release',\n",
    "        PROJECT_ROOT / 'Native' / 'pcai_core' / 'pcai_inference' / 'target-ffi-nosccache' / 'release',\n",
    "        Path('T:/RustCache/cargo-target/release')\n",
    "    ]\n",
    "    for base in [c for c in candidates if c]:\n",
    "        candidate = base / name\n",
    "        if candidate.exists():\n",
    "            return candidate\n",
    "    return None\n",
    "\n",
    "MODEL_PATH = find_model_path()\n",
    "LLAMACPP_BIN = find_binary('pcai-llamacpp.exe')\n",
    "MISTRAL_BIN = find_binary('pcai-mistralrs.exe')\n",
    "\n",
    "print('MODEL_PATH:', MODEL_PATH)\n",
    "print('LLAMACPP_BIN:', LLAMACPP_BIN)\n",
    "print('MISTRAL_BIN:', MISTRAL_BIN)\n",
    "\n",
    "if MODEL_PATH is None or LLAMACPP_BIN is None:\n",
    "    raise RuntimeError('Missing gguf model or pcai-llamacpp.exe. Set providers.pcai-native.modelPath in Config/llm-config.json and ensure binaries are in .local/bin.')\n",
]))

cells.append(md_cell([
    "## Parameter grid\n",
    "Tight defaults to keep runtime reasonable; expand for deeper sweeps.\n",
]))

cells.append(code_cell([
    "temperatures = [0.2, 0.8]\n",
    "max_tokens = [64, 128]\n",
    "gpu_layers = [-1, 0]\n",
    "datasets = ['diagnostic', 'safety']\n",
    "max_test_cases = 3\n",
    "\n",
    "backends = ['llamacpp-bin']\n",
    "if MISTRAL_BIN:\n",
    "    backends.append('mistralrs-bin')\n",
    "\n",
    "print('Backends:', backends)\n",
    "print('Datasets:', datasets)\n",
]))

cells.append(md_cell([
    "## Server control helpers\n",
    "We start a compiled server once per backend and GPU layer setting, then run all parameter sweeps against it (HTTP mode).\n",
]))

cells.append(code_cell([
    "def build_config(backend: str, model_path: Path, base_url: str, gpu_layers: int = -1):\n",
    "    resolved_model = model_path\n",
    "    if str(model_path).lower().endswith('.safetensors'):\n",
    "        resolved_model = model_path.parent\n",
    "\n",
    "    host, port = base_url.replace('http://', '').split(':')\n",
    "    config = {\n",
    "        'backend': { 'type': backend },\n",
    "        'model': {\n",
    "            'path': str(resolved_model),\n",
    "            'generation': {\n",
    "                'max_tokens': 512,\n",
    "                'temperature': 0.7,\n",
    "                'top_p': 0.95,\n",
    "                'stop': []\n",
    "            }\n",
    "        },\n",
    "        'server': { 'host': host, 'port': int(port), 'cors': True }\n",
    "    }\n",
    "    if backend == 'llama_cpp' and gpu_layers >= 0:\n",
    "        config['backend']['n_gpu_layers'] = gpu_layers\n",
    "    return config\n",
    "\n",
    "def wait_health(base_url: str, timeout_s: int = 60):\n",
    "    deadline = time.time() + timeout_s\n",
    "    while time.time() < deadline:\n",
    "        try:\n",
    "            with urllib.request.urlopen(base_url + '/health', timeout=3) as resp:\n",
    "                if resp.status == 200:\n",
    "                    return True\n",
    "        except Exception:\n",
    "            time.sleep(1)\n",
    "    return False\n",
    "\n",
    "def start_server(binary: Path, config_path: Path):\n",
    "    return subprocess.Popen([str(binary), '--config', str(config_path)], cwd=binary.parent)\n",
]))

cells.append(md_cell([
    "## System resource monitoring helper\n",
]))

cells.append(code_cell([
    "def start_resource_monitor(output_csv: Path, duration_s: int, refresh_s: int = 1):\n",
    "    ps = f'''\n",
    "Import-Module '{PROJECT_ROOT / 'Modules' / 'PC-AI.Performance' / 'PC-AI.Performance.psd1'}' -Force\n",
    "Watch-SystemResources -Duration {duration_s} -RefreshInterval {refresh_s} -OutputMode Object | Export-Csv -NoTypeInformation -Path '{output_csv}'\n",
    "'''\n",
    "    return subprocess.Popen([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', ps],\n",
    "                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)\n",
]))

cells.append(md_cell([
    "## Run evaluation grid\n",
    "Uses the PC-AI evaluation harness in HTTP mode against the compiled server.\n",
]))

cells.append(code_cell([
    "results = []\n",
    "\n",
    "for backend in backends:\n",
    "    binary = MISTRAL_BIN if backend == 'mistralrs-bin' else LLAMACPP_BIN\n",
    "    backend_type = 'mistral_rs' if backend == 'mistralrs-bin' else 'llama_cpp'\n",
    "\n",
    "    for server_gpu in gpu_layers:\n",
    "        port = get_free_port()\n",
    "        base_url = f'http://127.0.0.1:{port}'\n",
    "        config = build_config(backend_type, MODEL_PATH, base_url, gpu_layers=server_gpu)\n",
    "        config_path = REPORTS_DIR / f'config_{backend_type}_gpu{server_gpu}.json'\n",
    "        config_path.write_text(json.dumps(config, indent=2))\n",
    "\n",
    "        proc = start_server(binary, config_path)\n",
    "        if not wait_health(base_url, timeout_s=120):\n",
    "            proc.kill()\n",
    "            raise RuntimeError(f'Backend {backend} failed to start')\n",
    "\n",
    "        try:\n",
    "            for dataset in datasets:\n",
    "                for temp in temperatures:\n",
    "                    for max_tok in max_tokens:\n",
    "                        run_id = f'{backend}_{dataset}_t{temp}_mt{max_tok}_gpu{server_gpu}'.replace('.', 'p')\n",
    "                        out_json = REPORTS_DIR / f'{run_id}.json'\n",
    "                        out_csv = REPORTS_DIR / f'{run_id}_resources.csv'\n",
    "\n",
    "                        duration = max(20, max_test_cases * 5)\n",
    "                        monitor = start_resource_monitor(out_csv, duration_s=duration, refresh_s=1)\n",
    "\n",
    "                        cmd = f\"\"\"& '{PROJECT_ROOT / 'Tests' / 'Evaluation' / 'Invoke-InferenceEvaluation.ps1'}' `\n",
    "    -Backend http `\n",
    "    -BaseUrl {base_url} `\n",
    "    -ModelPath '{MODEL_PATH}' `\n",
    "    -Dataset {dataset} `\n",
    "    -MaxTokens {max_tok} `\n",
    "    -Temperature {temp} `\n",
    "    -GpuLayers {server_gpu} `\n",
    "    -MaxTestCases {max_test_cases} `\n",
    "    -OutputPath '{out_json}'\n",
    "\"\"\"\n",
    "                        print('Running:', run_id)\n",
    "                        ps_result = run_ps(cmd, check=False)\n",
    "\n",
    "                        try:\n",
    "                            monitor.wait(timeout=duration + 10)\n",
    "                        except subprocess.TimeoutExpired:\n",
    "                            monitor.kill()\n",
    "\n",
    "                        if ps_result.returncode != 0:\n",
    "                            print('Run failed:', run_id, ps_result.stderr)\n",
    "                            continue\n",
    "\n",
    "                        if not out_json.exists():\n",
    "                            print('Missing output:', out_json)\n",
    "                            continue\n",
    "\n",
    "                        payload = json.loads(out_json.read_text())\n",
    "                        summary = None\n",
    "                        for item in payload.get('Results', []):\n",
    "                            if item.get('Key') == 'http':\n",
    "                                summary = item.get('Value')\n",
    "                                break\n",
    "                        if summary is None:\n",
    "                            summary = {}\n",
    "\n",
    "                        results.append({\n",
    "                            'run_id': run_id,\n",
    "                            'backend': backend,\n",
    "                            'dataset': dataset,\n",
    "                            'temperature': temp,\n",
    "                            'max_tokens': max_tok,\n",
    "                            'gpu_layers': server_gpu,\n",
    "                            'pass_rate': summary.get('PassRate'),\n",
    "                            'avg_score': summary.get('AverageScore'),\n",
    "                            'avg_latency_ms': summary.get('AverageLatency'),\n",
    "                            'resources_csv': str(out_csv)\n",
    "                        })\n",
    "        finally:\n",
    "            proc.terminate()\n",
    "            try:\n",
    "                proc.wait(timeout=10)\n",
    "            except subprocess.TimeoutExpired:\n",
    "                proc.kill()\n",
    "\n",
    "df = pd.DataFrame(results)\n",
    "display(df.head())\n",
    "df.to_csv(REPORTS_DIR / 'summary.csv', index=False)\n",
    "print('Saved summary:', REPORTS_DIR / 'summary.csv')\n",
]))

cells.append(md_cell([
    "## Plot performance + quality metrics\n",
]))

cells.append(code_cell([
    "if df.empty:\n",
    "    raise RuntimeError('No results collected. Check logs in Reports for failures.')\n",
    "\n",
    "plt.figure(figsize=(12, 5))\n",
    "for backend in df['backend'].unique():\n",
    "    subset = df[df['backend'] == backend]\n",
    "    plt.plot(subset['avg_latency_ms'], label=f'{backend} latency')\n",
    "plt.title('Average Latency per Run')\n",
    "plt.xlabel('Run Index')\n",
    "plt.ylabel('Latency (ms)')\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "plt.figure(figsize=(12, 5))\n",
    "for backend in df['backend'].unique():\n",
    "    subset = df[df['backend'] == backend]\n",
    "    plt.plot(subset['avg_score'], label=f'{backend} score')\n",
    "plt.title('Average Quality Score per Run')\n",
    "plt.xlabel('Run Index')\n",
    "plt.ylabel('Score')\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]))

cells.append(md_cell([
    "## Plot system resource telemetry (CPU + memory)\n",
]))

cells.append(code_cell([
    "def plot_resources(csv_path):\n",
    "    data = pd.read_csv(csv_path)\n",
    "    if data.empty:\n",
    "        return\n",
    "    fig, ax1 = plt.subplots(figsize=(12, 5))\n",
    "    ax1.plot(data['CpuPercent'], label='CPU %')\n",
    "    ax1.set_ylabel('CPU %')\n",
    "    ax2 = ax1.twinx()\n",
    "    ax2.plot(data['MemoryPercent'], color='orange', label='Memory %')\n",
    "    ax2.set_ylabel('Memory %')\n",
    "    ax1.set_title(Path(csv_path).name)\n",
    "    fig.legend(loc='upper right')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "for csv_path in df['resources_csv'].head(3):\n",
    "    plot_resources(csv_path)\n",
]))

cells.append(md_cell([
    "## Optional: Standard LLM benchmark suite (lm-evaluation-harness)\n",
    "This step uses `lm-evaluation-harness` (EleutherAI) to run short subsets of common tasks (ARC Easy, HellaSwag, TruthfulQA). It is optional and will skip if installation fails.\n",
]))

cells.append(code_cell([
    "def ensure_lm_eval():\n",
    "    try:\n",
    "        import lm_eval  # noqa: F401\n",
    "        return True\n",
    "    except Exception:\n",
    "        try:\n",
    "            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lm-eval[openai]>=0.4.4'])\n",
    "            return True\n",
    "        except Exception as e:\n",
    "            print('lm-eval install failed:', e)\n",
    "            return False\n",
    "\n",
    "if ensure_lm_eval():\n",
    "    tasks = 'arc_easy,hellaswag,truthfulqa_mc'\n",
    "    base_url = 'http://127.0.0.1:8080/v1'\n",
    "    cmd = [\n",
    "        sys.executable, '-m', 'lm_eval',\n",
    "        '--model', 'openai-chat-completions',\n",
    "        '--model_args', f'model=pcai-inference,api_base={base_url},api_key=sk-local',\n",
    "        '--tasks', tasks,\n",
    "        '--limit', '5',\n",
    "        '--output_path', str(REPORTS_DIR / 'lm_eval_results.json')\n",
    "    ]\n",
    "    print('Running lm-eval:', ' '.join(cmd))\n",
    "    subprocess.run(cmd, check=False)\n",
    "else:\n",
    "    print('lm-eval not available; skipping standard benchmark suite.')\n",
]))

cells.append(md_cell([
    "## Optimization recommendation\n",
    "We compute a simple composite score to pick a high-performance configuration.\n",
]))

cells.append(code_cell([
    "df_clean = df.dropna(subset=['avg_score', 'avg_latency_ms']).copy()\n",
    "df_clean['score_norm'] = df_clean['avg_score']\n",
    "df_clean['lat_norm'] = 1.0 - (df_clean['avg_latency_ms'] / df_clean['avg_latency_ms'].max())\n",
    "df_clean['composite'] = (df_clean['score_norm'] * 0.6) + (df_clean['lat_norm'] * 0.4)\n",
    "best = df_clean.sort_values('composite', ascending=False).head(5)\n",
    "display(best)\n",
]))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

nb_path.write_text(json.dumps(nb, indent=2))
print(nb_path)
