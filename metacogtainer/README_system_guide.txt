--------------------------------------------------------------------------------------------------------

(a) First, you'll need to install Docker Desktop which you can find here: https://www.docker.com/

This will allow you to manage created Images and Containers easier.

(b) Then, you'll need to install Docker in your terminal, you can do that easily with: pip install docker

This will allow you to easily create and run Containers.

If you do not have the required libraries (tensorflow, numpy, mat73...):
	Navigate to the Dynamic-CFAR/metacogtainer folder
	Run the following command: pip install -r requirements.txt

If you are unable to run commands from the terminal, try starting Docker Desktop if you	have not already.

(c) Lastly, you'll need the code. Download the Dynamic-CFAR GitHub folder to your computer.

--------------------------------------------------------------------------------------------------------

(1) Open a terminal, such as Bash or Powershell, and navigate to the Dynamic-CFAR/metacogtainer folder

OPTIONAL: To make the code run faster, you can copy and run the following block of commands before running Step (2):

docker build -t metacog-base -f Dockerfile_base . ; 
docker build -t metacog-discriminator -f Dockerfile_disc . ; 
docker build -t metacog-evaluator -f Dockerfile_eval . ; 
docker build -t model-0 -f Model0_dockerfile . ;
docker build -t model-1 -f Model1_dockerfile . ;
docker build -t model-2 -f Model2_dockerfile . ;
docker build -t model-3 -f Model3_dockerfile . ;
docker build -t model-4 -f Model4_dockerfile . ;
docker build -t model-5 -f Model5_dockerfile . ;
docker build -t model-6 -f Model6_dockerfile . ;
docker build -t metacog-results -f Dockerfile_results . ;

(2) Run the following command: python python-setup-script.py

(3) The output will appear in the terminal after all the containers have run.
    For explanations of the commands, please view setup_commands_all.txt
    
    Docker MetaCog Code.pdf contains explanations for changes made to the python code  - OUTDATED
    Docker MetaCog Containers.pdf contains explanations for the overall Container system - OUTDATED

--------------------------------------------------------------------------------------------------------