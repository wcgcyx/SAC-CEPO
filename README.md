# SAC-CEPO
Code and documentation around benchmarking Soft Actor-Critic with Cross-Entropy Policy Optimization (SAC-CEPO) against the original Soft Actor-Critic (SAC).

## Pre-requistites (Windows 10)
### Install python
Python 3.7.6 is recommended. Make sure 64-bit version is installed.
https://www.python.org/downloads/release/python-376/
### Install pytorch
Install pytorch 1.4.0 by
`pip3 install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html`
To run the code with GPU (recommended):
1. Update GPU driver to be 418.x or higher
https://www.nvidia.com/download/index.aspx?lang=en-us
2. Install CUDA 10.1
https://developer.nvidia.com/cuda-10.1-download-archive-update2
3. Install cuDNN (version >= 7.6) for CUDA 10.1
https://developer.nvidia.com/cudnn
### Install Mujoco
1. Create folder _**.mujoco**_ under _**%userprofile%**_
2. Create folder _**.mujoco/mujoco200**_
3. Download mujoco200 win64 from:
https://www.roboti.us/index.html
4. Extract all files into _**.mujoco/mujoco200**_ so that bin files can be found under _**.mujoco/mujoco200/bin/**_
5. Mujoco requires a licence to run. A free 30-days trail licence can be obtained through:
https://www.roboti.us/license.html
6. Copy the licence _**mjkey.txt**_ to _**.mujoco/mjkey.txt**_
7. Add _**.mujoco/mujoco200/bin**_ to PATH.
### Install Microsoft Visual C++ build tools 2019
Download Build Tools for Visual Studio 2019 from:
https://visualstudio.microsoft.com/downloads/
When installing, make sure **C++ build tools** is selected.
### Install openAI Gym
1. Install gym via pip by `pip3 install gym`
2. Install cffi and pygit2 by `pip3 install cffi pygit2`
3. Clone mujoco-py repository by `git clone https://github.com/openai/mujoco-py.git` (Install git first if not installed)
4. `cd mujoco-py`
5. `py -3 -m pip install --upgrade setuptools`
6. `pip3 install -r requirements.txt`
7. `pip3 install -r requirements.dev.txt`
8. Open _**\mujoco-py\scripts\gen_wrappers.py**_,  _**\mujoco-py\mujoco_py\generated\wrappers.pxi**_ and replace all instances of **isinstance(addr, (int, np.int32, np.int64))** with **hasattr(addr, '__int__')**
9. `cd /mujoco-py/`
10. Compile mujoco_py by `python -c "import mujoco_py`
11. Install mujoco-py by `py -3 setup.py install`

## Quick start
* To run SAC with Pendulum environment
`py -3 train.py sac Pendulum-v0 5 10000 test.csv`
* To run SAC-CEPO with Pendulum environment
`py -3 train.py cepo Pendulum-v0 5 10000 test.csv`

## Contributor
* Zhenyang Shi - University of Queensland (z.shi@uqconnect.edu.au)

## Note
**Please do NOT distribute/modify the code at this stage.**