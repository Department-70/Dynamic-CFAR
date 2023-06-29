--------------------------------------------------------------------------------------------------------

(a) First, you'll need to install Docker Desktop which you can find here: https://www.docker.com/

This will allow you to manage created Images and Containers easier.

(b) Then, you'll need to install Docker in your terminal, you can do that easily with: pip install docker

This will allow you to easily create and run Containers.

If you are unable to run commands from the terminal, try starting Docker Desktop if you	have not already.

(c) Lastly, you'll need the code. Download the Dynamic-CFAR GitHub folder to your computer.

--------------------------------------------------------------------------------------------------------

(1) Open a terminal, such as Bash or Powershell, and navigate to the Dynamic-CFAR/metacogtainer folder

OPTIONAL: To make the code run faster, you can copy and run the following block of commands before running Step (2):

docker build -t metacog-base -f Dockerfile_base . ; 
docker build -t metacog-discriminator -f Dockerfile_disc . ; 
docker build -t metacog-evaluator -f Dockerfile_eval . ; 
docker build -t metacog-fa-cd -f Dockerfile_fa_cd . ; 
docker build -t metacog-fa-glrt -f Dockerfile_fa_glrt . ; 
docker build -t metacog-fa-ideal -f Dockerfile_fa_ideal . ; 
docker build -t metacog-results -f Dockerfile_results .

(2) Run the following command: python python-setup-script.py

(3) The output will appear in the terminal after all the containers have run.
    For explanations of the commands, please view system_setup_commands.txt
    
    Docker MetaCog Code.pdf contains explanations for changes made to the python code
    Docker MetaCog Containers.pdf contains explanations for the overall Container system

--------------------------------------------------------------------------------------------------------