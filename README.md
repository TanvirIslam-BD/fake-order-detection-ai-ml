# Fake order prediction using Machine Learning


## Running locally
To run the basic server, you'll need to install a few requirements. To do this, run:

```bash
pip install -r requirements/common.txt
```

This will install only the dependencies required to run the server. 

### Creating a model

Before deploying a model, you'll first need to train a model. There's a demo pipeline
setup in this project. To train the model, run:

```bash
python steps/train.py --path=data/heart-disease.csv 
```

This'll produce a couple of files in the `/data` directory, including a pickled 
`Pipeline` object. Now you can deploy this pipeline as an API!

### Launching the API

To boot up the default server, you can run:

```bash
bash bin/run.sh
```

This will start a [Gunicorn](https://gunicorn.org/) server that wraps the Flask app 
defined in `src/app.py`. Note that [this is one of the recommended ways of deploying a
Flask app 'in production'](https://flask.palletsprojects.com/en/1.1.x/deploying/wsgi-standalone/). 
The server shipped with Flask is [intended for development
purposes only](https://flask.palletsprojects.com/en/1.1.x/deploying/#deployment).  

You should now be able to send:

```bash
curl localhost:5000/health
```

And receive the response `OK` and status code `200`. 

### Querying the model

You can now query the model using:

```bash
curl --location --request POST '127.0.0.1:5000/predict' \
--header 'Content-Type: application/json' \
-d @data/payload.json
```

Where the payload looks like:

```json
{
    "sex": 0,
    "cp": 1,
    "restecg": 0,
    "ca": 1,
    "slope": 1,
    "thal": 2,
    "age": 57,
    "trestbps": 130,
    "chol": 236,
    "fbs": 0,
    "thalach": 174,
    "exang": 0,
    "oldpeak": 0.0
}
```

You should see a response looking something like:

```json
{
    "diagnosis": "heart-disease"
}
```

And that's it, you have a model wrapped in a web API!

## Running with `docker`

Unsurprisingly, you'll need [Docker](https://www.docker.com/products/docker-desktop) 
installed to run this project with Docker. To build a containerised version of the API, 
run:

```bash
docker build . -t flask-app
```

To launch the containerised app, run:

```bash
docker run -p 8080:8080 flask-app
```

You should see your server boot up, and should be accessible as before!

With that done, you've got a containerised ML web API ready to go.

## Developing with the template

To develop the template for your own project, you'll need to make sure to [create your
own repository from this template](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template) 
and then install the project's development dependencies. You can do this with:

```bash
pip install -r requirements/develop.txt
```

This'll install some style formatting and testing tools (including `pytest` and 
`locust`).
