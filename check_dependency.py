import importlib.util
import subprocess
import os

def get_cuda_version():
    """Attempts to find the system CUDA Toolkit version."""
    # Method 1: Check nvcc (The most reliable for the Toolkit)
    try:
        out = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        if "release" in out:
            return out.split("release ")[1].split(",")[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Method 2: Check nvidia-smi (Driver-based CUDA version)
    try:
        out = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        if "CUDA Version:" in out:
            return out.split("CUDA Version:")[1].split("|")[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Method 3: Check common installation paths
    cuda_path = "/usr/local/cuda/version.txt"
    if os.path.exists(cuda_path):
        with open(cuda_path, 'r') as f:
            return f.read().replace("CUDA Version", "").strip()

    return "Not Found"

def check_h5py_mpi():
    try:
        import h5py
        return "✅ Enabled" if h5py.get_config().mpi else "❌ Serial Only"
    except ImportError:
        return "⚠️ Not Installed"

def check_dependencies(packages):
    print(f"{'Package':<20} | {'Status':<15} | {'Version'}")
    print("-" * 55)

    for pkg in packages:
        # Specialized handling for system-level or complex packages
        if "cuda-toolkit" in pkg.lower():
            version = get_cuda_version()
            status = "✅ Found" if version != "Not Found" else "⚠️ Missing"
            print(f"{pkg:<20} | {status:<15} | {version}")
            continue

        is_h5py_mpi = "h5py" in pkg.lower() and "mpi" in pkg.lower()
        search_name = "h5py" if is_h5py_mpi else pkg.split('(')[0].strip().lower().replace('-', '_')
        
        # Mapping for inconsistent naming
        mapping = {"parmed": "parmed", "openmm": "openmm"}
        actual_import = mapping.get(search_name, search_name)
        
        spec = importlib.util.find_spec(actual_import)
        if spec is not None:
            try:
                module = importlib.import_module(actual_import)
                version = getattr(module, '__version__', 'Installed')
                status = check_h5py_mpi() if is_h5py_mpi else "✅ Found"
            except ImportError as e:
                status = "❌ Link Err"
                version = str(e).splitlines()[0][:20]
        else:
            status = "⚠️ Missing"
            version = "N/A"
            
        print(f"{pkg:<20} | {status:<15} | {version}")

groups = {
    "Basic": ["numpy", "scipy", "psutil", "pyscf", "mpi4py", "h5py (mpi-io)"],
    "GPU Support": ["cuda-toolkit", "cupy", "gpu4pyscf"],
    "QM/MM": ["openmm", "parmed"]
}

for group, libs in groups.items():
    print(f"\n--- {group} ---")
    check_dependencies(libs)
