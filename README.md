# Dependencies

Please refer to the [`user manual`](http://doc.lonxun.com/PWMLFF/active%20learning/)

1. AL-PWMLFF job scheduling uses the [SLURM](https://slurm.schedmd.com/documentation.html) cluster management and job scheduling system. SLURM must be installed on your computing cluster.

2. DFT calculations in AL-PWMLFF support [PWmat](https://www.pwmat.com/gpu-download), [VASP](https://www.vasp.at/), [CP2K](https://www.cp2k.org/) and DFTB. We have integrated DFTB in PWmat. You can find detailed usage instructions in the `DFTB_DETAIL section` of the [`PWmat Manual`](http://www.pwmat.com/pwmat-resource/Manual.pdf).

3. AL-PWMLFF model training is based on [`PWMLFF`](https://github.com/LonxunQuantum/PWMLFF). Refer to the [`PWMLFF documentation`](http://doc.lonxun.com/PWMLFF/Installation) for installation instructions ([`Download address for PWmat version integrated with DFTB`](https://www.pwmat.com/modulefiles/pwmat-resource/mstation-download/cuda-11.6-mstation-beta.zip)).

4. AL-PWMLFF Lammps molecular dynamics simulation is based on [Lammps_for_pwmlff](https://github.com/LonxunQuantum/Lammps_for_PWMLFF/tree/libtorch). Refer to the [`Lammps_for_pwmlff documentation`](https://github.com/LonxunQuantum/Lammps_for_PWMLFF/blob/libtorch/README) for installation instructions.

# Installation Process
You can install it through the pip command or the github source code installation.

## install by pip

```bash
    pip install pwact
```
## from github

### Code Download

    git clone https://github.com/LonxunQuantum/PWact.git

Then import environment variable.

```
export PATH=/data/home/wuxingxing/codespace/al_pwmlff/bin:$PATH
```
AL-PWMLFF is developed in Python and supports Python 3.9 and above. It is recommended to use the Python runtime environment provided by PWMLFF.

If you need to create a virtual environment for AL-PWMLFF separately, you only need to install the following dependent packages (compatible with your Python version, Python >= 3.9).
```bash
pip install numpy pandas tqdm pwdata
```

    
# Command List

AL-PWMLFF includes the following commands, which are not case sensitive. The starting command is `pwact`

### 1. Display the available command list

```bash
pwact  [ -h / --help / help ]
```

### 2. Display the parameter list for cmd_name:

```bash
pwact cmd_name -h
```

### 3. Initial Training Set Preparation

```bash
pwact init_bulk param.json resource.json
```

### 4. Active Learning

```bash
pwact run param.json resource.json
```

For the 3-th and 4-th command above, the names of the JSON files can be modified by the user, but it is required that the input order of [`param.json`](#paramjson) and [`resouce.json`](#resourcejson) cannot be changed.

### 5. Tool Commands

Convert MOVEMENT or OUTCAR to PWdata format

```bash
pwact to_pwdata
```

Search for labeled datasets in the active learning directory

```bash
pwact gather_pwdata
```

### examples download
from github
``` 
https://github.com/LonxunQuantum/PWact/tree/main/pwact/example
```

from BaiduNetdisk included the calculation results of examples
```
https://pan.baidu.com/s/14E0u_7cpntiBZgg-C1S5XA?pwd=pwmt
```
