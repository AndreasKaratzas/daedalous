# Rainbow DQN for Autonomous Drone Navigation


[//]: # (There is a bug related to PyLint. The `src` directory is not properly recognized and therefore PyLint cannot find the imported files throughout the project. To temporarily resolve this issue, set "python.analysis.autoSearchPaths": false .)

[//]: # (1. pip install msgpack-rpc-python 2. pip install airsim 3. pip install -e envs)


1. INSTALL VS 2019 COMMUNITY FOR C++ DEVELOPMENT (PACKAGES)
2. INSTALL CMAKE LATEST VERSION FROM MSI FILE
3. INSTALL ANACONDA 
4. CREATE PROJECT DIRECTORY
5. SETUP PYTHON
	- RUN ANACONDA DEVELOPER - OR UPDATE IT (conda update -n base conda) - AS ADMIN
	- CHANGE DIRECTORY (cd "C:\Users\giann\OneDrive\Έγγραφα\DAEDALOUS")
	- CREATE A NEW CONDA ENVIRONMENT (conda create --name daedalous python=3.8)
	- ACTIVATE IT (conda activate daedalous)
	- INSTALL PACKAGES - THE ORDER IS IMPORTANT !!!!
		* `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
		* `conda install -c conda-forge opencv`
		* `conda install numpy`
		* `conda install -c conda-forge matplotlib`
		* `python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose`
		* `conda install -c conda-forge gym`
		* `pip install msgpack-rpc-python`
		* `pip install airsim`
	- INSTALL CUSTOM ENV (pip install -e gym-daedalous)
6. OPEN VS 2019 COMMUNITY AS ADMIN
7. GO TO TOOLS -> COMMAND LINE -> DEVELOPER COMMAND PROMPT
8. DOWNLOAD AIRSIM 1.4 (https://github.com/microsoft/AirSim/releases/tag/v1.4.0-windows) ZIP FILE
9. DOWNLOAD THE 3 CITYENVIRON ZIP FILES
10. DOWNLOAD 7-ZIP TO EXTRACT THOSE FILES
11. CHANGE DIRECTORY INTO AIRSIM FOLDER WITH VS 2019 DEVELOPER COMMAND PROMPT (cd "C:\Users\giann\OneDrive\Έγγραφα\DAEDALOUS\AirSim")
12. BUILD AIRSIM (build.cmd)
13. REPLACE SETTINGS IN FILE %USER/AIRSIM/SETTINGS.JSON WITH THE PROJECT SETTINGS FOUND UNDER (...)
14. EXTRACT CITY ENVOIRONMENT AND PASTE THE RUN.BAT INTO THE PARENT FOLDER OF THE EXECUTABLE
15. OPEN WINDOWS TERMINAL IN THAT DIRECTORY AND RUN THE RUN.BAT SCRIPT
16. GO BACK TO THE ANACONDA WINDOW, EXECUTE `CLEAR`, AND THEN `PYTHON DAEDALOUS.PY`

X. TODO : ADD TEST MODE
