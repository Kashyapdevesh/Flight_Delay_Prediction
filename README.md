## Flight_Delay_Prediction

# Usage of the API  

This example considers that the API was launched locally without docker and 
with the default parameters (`localhost` at port `5000`) and its calling 
the example model.

# Docker commands 
Note: Docker tag or id should be always specified in the end of the docker command to avoid issues
1. Build docker image from Dockerfile

    ```docker build -t "<app name>" -f docker-files/Dockerfile .```
    ```eg: docker build -t "ml_app" -f docker-files/Dockerfile .```

2. Run the docker container after build

    ```docker run -p 9999:9999 ml_app  # -p to make the port externally avaiable for browsers```

3. Show all running containers
    
    ```docker ps```

    a. Kill and remove running container
    
     ```docker rm <containerid> -f ```

4. Open bash in a running docker container (optional)

    ```docker exec -ti <containerid> bash```
5. Docker Entry point
The ENTRYPOINT specifies a command that will always be executed when the container starts. The CMD specifies arguments that will be fed to the ENTRYPOINT
1683

Docker has a default ENTRYPOINT which is /bin/sh -c but does not have a default CMD.
--entrypoint in docker run will overwrite the default entry point
    ```docker run -it --entrypoint /bin/bash <image>```


### Check the API's health status

Endpoint: `/health`

```bash
$ curl -X GET http://localhost:5000/health
up
```

### Is model ready?

Endpoint: `/ready`

```bash
$ curl -X GET http://localhost:5000/ready
ready
```


### Compute predictions

Endpoint: `/predict`

```bash

```

