docker run \
	-e GRANT_SUDO=yes -e NB_UID=$UID -e NB_GID=$UID \
	--gpus all \
	-itd \
	--shm-size=16gb \
	-p $2:$2 \
	--name $1 \
	--mount type=bind,source=/home,target=/home \
	--mount type=bind,source=/mnt,target=/mnt \
	pytorch:v1.4 \
	fish

docker exec -it $1 fish

