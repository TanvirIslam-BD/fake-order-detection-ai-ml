# Genuine/Fake order prediction using Machine Learning


## Running locally
To run the basic server, you'll need to install a few requirements. To do this, run:

```bash
pip install -r requirements/common.txt
```

This will install only the dependencies required to run the server. 

This'll produce a couple of files in the `/data` directory, including a pickled 
`Pipeline` object. Now you can deploy this pipeline as an API!

### Launching the APP

To boot up the default server, you can run:

```bash
bash bin/run.sh
```

This will start a [Gunicorn](https://gunicorn.org/) server that wraps the Flask app 
defined in `src/app.py`. Note that [this is one of the recommended ways of deploying a
Flask app 'in production'](https://flask.palletsprojects.com/en/1.1.x/deploying/wsgi-standalone/). 
The server shipped with Flask is [intended for development
purposes only](https://flask.palletsprojects.com/en/1.1.x/deploying/#deployment).  


### Querying the model

You can now query the model using:

```bash
curl --location --request POST '127.0.0.1:5000/api/v1/predict' \
--header 'Content-Type: application/json' \
-d @data/payload.json
```

Where the payload looks like:

```json
{
  "Amount(Total Price)": 100,
  "Country": "Australia"
}
```

You should see a response looking something like:

```json
{
    "prediction": "Genuine Order"
}
```

And that's it, you have a model wrapped in a web app!
## Developing with the template

To develop the template for your own project, you'll need to make sure to [create your
own repository from this template](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template) 
and then install the project's development dependencies. You can do this with:

```bash
pip install -r requirements/develop.txt
```

This'll install some style formatting and testing tools (including `pytest` and`locust`).
