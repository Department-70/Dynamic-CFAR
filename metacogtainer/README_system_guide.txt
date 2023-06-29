--------------------------------------------------------------------------------------------------------

(a) First, you'll need to install Docker Desktop which you can find here: https://www.docker.com/

This will allow you to manage created Images and Containers easier.

(b) Then, you'll need to install Docker in your terminal, you can do that easily with: pip install docker

This will allow you to easily create and run Containers.

If you are unable to run commands from the terminal, try starting Docker Desktop if you	have not already.

(c) Lastly, you'll need the code. Clone the Dynamic-CFAR GitHub folder to your computer.

--------------------------------------------------------------------------------------------------------

(1) Open a terminal, such as Bash or Powershell, and navigate to the Dynamic-CFAR/metacogtainer folder

(2) Copy and run the following command:

docker volume create metacog-volume 

(3) Copy and run the following block of commands:

docker build -t metacog-base -f Dockerfile_base . ; 
docker build -t metacog-discriminator -f Dockerfile_disc . ; 
docker build -t metacog-evaluator -f Dockerfile_eval . ; 
docker build -t metacog-fa-cd -f Dockerfile_fa_cd . ; 
docker build -t metacog-fa-glrt -f Dockerfile_fa_glrt . ; 
docker build -t metacog-fa-ideal -f Dockerfile_fa_ideal . ; 
docker build -t metacog-results -f Dockerfile_results .

(4) Copy and run the following block of commands:

docker run --mount type=volume,source="metacog-volume",target=/app/docker_bind metacog-discriminator ;
docker run --mount type=volume,source="metacog-volume",target=/app/docker_bind metacog-evaluator ;
docker run --mount type=volume,source="metacog-volume",target=/app/docker_bind metacog-fa-cd ;
docker run --mount type=volume,source="metacog-volume",target=/app/docker_bind metacog-fa-glrt ;
docker run --mount type=volume,source="metacog-volume",target=/app/docker_bind metacog-fa-ideal ;
docker run --mount type=volume,source="metacog-volume",target=/app/docker_bind metacog-results

(5) The output will appear in the terminal after all the containers have run.
    For explanations of the commands, please view system_setup_commands.txt
    
    Docker MetaCog Code.pdf contains explanations for changes made to the python code
    Docker MetaCog Containers.pdf contains explanations for the overall Container system

--------------------------------------------------------------------------------------------------------