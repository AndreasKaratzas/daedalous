# Rainbow DQN for Autonomous Drone Navigation

[//]: # (There is a bug related to PyLint. The `src` directory is not properly recognized and therefore PyLint cannot find the imported files throughout the project. To temporarily resolve this issue, set "python.analysis.autoSearchPaths": false .)

[//]: # (1. pip install msgpack-rpc-python 2. pip install airsim 3. pip install -e envs)


# Prerequisites

1. Install VS 2019 Community for C++ development.
2. Install `cmake` latest version from `.msi` file.
3. (Optional) Download and install `7zip`.

# Installation

1. Launch a terminal instance with administrator rights
2. Create a virtual environment using `python -m venv daedalous`.
3. Activate the new python virtual interpreter using:
	* `source daedalous/bin/activate` for Unix users.
	* `.\daedalous\Scripts\activate` for Windows users.
4. Upgrade it using `python -m pip install --upgrade pip`.
5. Install all required packages in this **exact** order:
	* `python -m pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html`
	* `python -m pip install opencv-python`
	* `python -m pip install numpy scipy matplotlib ipython jupyter pandas sympy nose`
	* `python -m pip install gym`
	* `python -m pip install msgpack-rpc-python`
	* `python -m pip install airsim`
	* `python -m pip install -e envs`
6. Launch the VS 2019 command prompt with administrator rights.
7. Download `AirSim 1.6` and the 3 `CityEnviron` compressed files.
8. Extract `CityEnviron` files.
9. Use the VS 2019 command prompt to navigate to the `AirSim` directory and build it by executing `.\build.cmd`
10. Adjust the settings file under `~\Documents\AirSim\settings.json`. The recommended settings are at `airsim_settings.json`.
11. Navigate in the `CityEnviron` directory and paste the `run.bat` file.
12. Launch a terminal instance and execute the `run.bat` script.
13. Launch a second terminal instance and execute `python .\main.py`.

# Notes

Tested under Windows 10 OS.

# TODOs

1. Add test mode.
2. Add explainability using [TruLens](https://www.trulens.org/?fbclid=IwAR2ZMGETnpzAj2Nxt9RVmGrSxDDGkb3kaMsIgHJqRZ4aNqH5t4gATsBTfw4)
