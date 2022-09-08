export DOCKER_BUILDKIT=1


report_failure() {
    echo "FAILED on line $1"
    exit 255
}

for var in "$@"
do
    aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/i9g5h9s5
    docker build -t $var ./$var
    docker tag $var:latest public.ecr.aws/i9g5h9s5/$var:latest
    docker push public.ecr.aws/i9g5h9s5/$var:latest
done

trap 'report_failure $LINENO' ERR