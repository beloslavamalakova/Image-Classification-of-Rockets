docker-build:
	docker build . -f Dockerfile -t 

run:
	docker run -it --net=host -v `pwd`:/ Image-Classification-of-Rockets /bin/bash

run-gpu:
	docker run --gpus all -it --net=host -v `pwd`:/Image-Classification-of-Rockets Image-Classification-of-Rockets/bin/bash