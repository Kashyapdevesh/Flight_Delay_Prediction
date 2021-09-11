## Flight_Delay_Prediction

# Usage of the API  

This example considers that the API was launched locally without docker and 
with the default parameters (`localhost` at port `5003`) and its calling 
the example model.

# Docker commands 
Note: Docker tag or id should be always specified in the end of the docker command to avoid issues
1. Build docker image from Dockerfile

    ```docker build -t "<app name>" -f ./Dockerfile .```
    ```eg: docker build -t "ml_app" -f ./Dockerfile .```

2. Run the docker container after build

    ```docker run -p 5003 ml_app  # -p to make the port externally avaiable for browsers```

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

### Wiping models after use (NOTICE: Future Scope for multi- model approach)

Endpoint: `/wipe`

```bash
$ curl -X GET http://localhost:5000/wipe
Model wiped
```

While all the above API requests can be accesed via simple HTTP 'GET' request, the prediction model requires input in JSON format accesible through 'POST' request as shown below :

```bash
[
    {
        "iataCode": "ATL",
        "date": "2021-09-12",
        "time": "16:25"
    },
    {
        "iataCode": "JFK",
        "date": "2021-09-13",
        "time": "12:30"
    }
]


```
The API will provide output prediction in the same JSON format as shown below:

```bash
{
    "Prediction": "45",
    "Prediction": "12"
}

```
Moreover, we have also provided provison for bulk predictions thorugh the same JSON requests.
