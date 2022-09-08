#!/bin/bash

source .yml.env
docker-compose up --detach

echo
echo If you want to see logs, run :
echo docker-compose logs -f
