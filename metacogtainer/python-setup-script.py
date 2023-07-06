import docker
# pip install cryptography==36.0.2

# Gets access to Docker from Python
client = docker.from_env()

print("\n------------------------------------\n")

# Creates Volume for long term storage
print("MetaCog File Volume Ready...", )
client.volumes.create(name='metacog-volume')

# Checks if Docker images exist, if not they are built before continuing
try: 
    client.images.get('metacog-base')
except docker.errors.ImageNotFound:
    print("Building MetaCog Base...")
    client.images.build(path='.', tag='metacog-base', dockerfile='Dockerfile_base')

print("MetaCog Base Image Ready...")

try: 
    client.images.get('metacog-discriminator')
except docker.errors.ImageNotFound:
    print("Building MetaCog Discriminator...")
    client.images.build(path='.', tag='metacog-discriminator', dockerfile='Dockerfile_disc')
    
try: 
    client.images.get('metacog-evaluator')
except docker.errors.ImageNotFound:
    print("Building MetaCog Evaluator...")
    client.images.build(path='.', tag='metacog-evaluator', dockerfile='Dockerfile_eval')
    
try: 
    client.images.get('model-0')
except docker.errors.ImageNotFound:
    print("Building Model-0...")
    client.images.build(path='.', tag='model-0', dockerfile='Model0_dockerfile')

try: 
    client.images.get('model-1')
except docker.errors.ImageNotFound:
    print("Building Model-1...")
    client.images.build(path='.', tag='model-1', dockerfile='Model1_dockerfile')

try: 
    client.images.get('model-2')
except docker.errors.ImageNotFound:
    print("Building Model-2...")
    client.images.build(path='.', tag='model-2', dockerfile='Model2_dockerfile')

try: 
    client.images.get('model-3')
except docker.errors.ImageNotFound:
    print("Building Model-3...")
    client.images.build(path='.', tag='model-3', dockerfile='Model3_dockerfile')

try: 
    client.images.get('model-4')
except docker.errors.ImageNotFound:
    print("Building Model-4...")
    client.images.build(path='.', tag='model-4', dockerfile='Model4_dockerfile')

try: 
    client.images.get('model-5')
except docker.errors.ImageNotFound:
    print("Building Model-5...")
    client.images.build(path='.', tag='model-5', dockerfile='Model5_dockerfile')

try: 
    client.images.get('model-6')
except docker.errors.ImageNotFound:
    print("Building Model-6...")
    client.images.build(path='.', tag='model-6', dockerfile='Model6_dockerfile')
    
try: 
    client.images.get('metacog-results')
except docker.errors.ImageNotFound:
    print("Building MetaCog Results...")
    client.images.build(path='.', tag='metacog-results', dockerfile='Dockerfile_results')

print("\n---------- Setup Complete ----------\n")


# Runs each container sequentially
print("Running discriminator...")
client.containers.run('metacog-discriminator', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("Running evaluator...")
client.containers.run('metacog-evaluator', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("Reading data with Model-0...")
client.containers.run('model-0', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("Reading data with Model-1...")
client.containers.run('model-1', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("Reading data with Model-2...")
client.containers.run('model-2', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("Reading data with Model-3...")
client.containers.run('model-3', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("Reading data with Model-4...")
client.containers.run('model-4', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("Reading data with Model-5...")
client.containers.run('model-5', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("Reading data with Model-6...")
client.containers.run('model-6', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("\n--------- Process Complete ---------\n")

print("Gathering results...\n")
results = client.containers.run('metacog-results', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}}, stdout=True)
print( results.decode() ) # leaves out \n from string