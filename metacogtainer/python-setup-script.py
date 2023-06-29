import docker
# pip install cryptography==36.0.2

# Gets access to Docker from Python
client = docker.from_env()

# Creates Volume for long term storage
print("Creating MetaCog Volume...", )
client.volumes.create(name='metacog-volume')

# Checks if Docker images exist, if not they are built before continuing
try: 
    client.images.get('metacog-base')
except docker.errors.ImageNotFound:
    print("Building MetaCog Base...")
    client.images.build(path='.', tag='metacog-base', dockerfile='Dockerfile_base')

print("MetaCog Base Ready...")

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
    client.images.get('metacog-fa-cd')
except docker.errors.ImageNotFound:
    print("Building MetaCog FA-CD...")
    client.images.build(path='.', tag='metacog-fa-cd', dockerfile='Dockerfile_fa_cd')
    
try: 
    client.images.get('metacog-fa-glrt')
except docker.errors.ImageNotFound:
    print("Building MetaCog FA-GLRT...")
    client.images.build(path='.', tag='metacog-fa-glrt', dockerfile='Dockerfile_fa_glrt')
    
try: 
    client.images.get('metacog-fa-ideal')
except docker.errors.ImageNotFound:
    print("Building MetaCog FA-IDEAL...")
    client.images.build(path='.', tag='metacog-fa-ideal', dockerfile='Dockerfile_fa_ideal')
    
try: 
    client.images.get('metacog-results')
except docker.errors.ImageNotFound:
    print("Building MetaCog Results...")
    client.images.build(path='.', tag='metacog-results', dockerfile='Dockerfile_results')
        
# Runs each container sequentially
print("Running discriminator...")
client.containers.run('metacog-discriminator', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("Running evaluator...")
client.containers.run('metacog-evaluator', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("Running FA-CD...")
client.containers.run('metacog-fa-cd', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("Running FA-GLRT...")
client.containers.run('metacog-fa-glrt', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("Running FA-IDEAL...")
client.containers.run('metacog-fa-ideal', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}})

print("--- Process Complete ---")

print("Gathering results...")
results = client.containers.run('metacog-results', volumes={'metacog-volume': {'bind': '/app/docker_bind', 'mode': 'rw'}}, stdout=True)
print( results[0:-1] ) # leaves out \n from string