--------------------------------------------------------------------------------------------------------

(a) First, you'll need to install Docker Desktop which you can find here: https://www.docker.com/

This will allow you to manage created Images and Containers easier.

(b) Then, you'll need to install Docker in your terminal, you can do that easily with: pip install docker

This will allow you to easily create and run Containers.

If you are unable to run commands from the terminal, try starting Docker Desktop if you	have not already.

(c) Lastly, you'll need the code. Download the Dynamic-CFAR GitHub folder to your computer.

NOTE:   Only copy the required data file in the ./Datasets folder,
	This is because the whole folder will be copied into each container.
	Thus, if you have all of them you could be copying 20+ gigabytes of data into all the containers.

NOTE: 	You do not need to install ./requirements.txt,
	These files are needed inside the Docker Containers and are installed automatically.
	The MetaCog Base Image handles this and is automatically run by Docker Compose.

--------------------------------------------------------------------------------------------------------

(1) Open a terminal, such as Bash or Powershell, and navigate to the Dynamic-CFAR/metacogtainer folder

OPTIONAL: To make the code run faster, you can copy and run the following block of commands before running Step (2):
	  This builds the Images more efficiently, when building in Compose / Python SDK it can take hours.
	  With these commands, it should take under ten minutes.

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
docker build -t metacog-glrt-f Dockerfile_glrt . 

(2) Run the following command:  docker compose up

(3) The output will appear in the terminal after all the containers have run.

    For explanations on how to run containers individually, please view setup_commands_all.txt

(4) To close the system run the following command: docker compose down

--------------------------------------------------------------------------------------------------------