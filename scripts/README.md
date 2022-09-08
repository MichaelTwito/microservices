### Run Scripts
## Local
Run the apps on local machine, make sure to build venv first
```
./run_local.sh
```
## Docker-Compose
```
./run_docker_compse.sh
```

### Deployment Scripts
## Release to ecr
Builds the containers according to Dockerfile, then push them into ECR,
(you can choose to run only one app)
```
./release_to_ecr prediction_manger dolores
```
## Seed Dataset to EFS
Seeds a dataset into EFS trought EC2 instance by SCP
```
./release_to_ecr -k <private_key_for_ec2_connection> -d <path_to_dataset> -r <user@ip_addr_of_ec2_instance> 
```
