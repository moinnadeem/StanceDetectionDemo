# Demo

This demo is used to serve our state of the art stance detection model as tested on the Fake News Challenge.
The details about the model are discussed in a short paper which is linked [here]{https://marcpickett.com/cl2018/CL-2018_paper_29.pdf}.
These results were presented at the 2018 NIPS Continual Learning Workshop.

## Code Structure

The file run_saved_model.py contains the code used to load the saved model and serve predictions for arbitrary
results. In order to load the model, the following pickle data related to that model is saved in
pickle_data/fnc_fever_3/.

- bow_vectorizer.pickle
- embedding_matrix.npy
- tfidf_vectorizer.pickle
- tfreq_vectorizer.pickle
- word_index.npy

Because this information, particularly the embedding_matrix, is fairly large, it cannot be saved directly into the github repository.

Using a Flask framework a simple site is used to retrieve user input regarding the claim and document, run that
input through the saved model, and serve an output page with the results. All of the Flask related code is
saved in the app/ folder. The routing is controlled in the index() function in app/routing.py. The page templates 
are saved in app/templates/input.html and app/templates/output.html. CSS for these pages is saved in app/static/css/main.css.

## Running the Demo

Server currently running on beorn machine.

Run the app using the command

```
flask run --host=0.0.0.0
```

setting the host to 0.0.0.0 allows other computers to access the app.

After running the server, you should be able to see it at 

``` 
machine-name:port
```   

Normally the port number is 5000 by default. So, if you run it on the beorn machine it will be 

```   
beorn.csail.mit.edu:5000
```   

Note that some SLS machines don't allow remote connections, such as sls-quad-30, so use a different one.

## Requirements

A subset of the following libraries is required to run the code. The list below was procured using pip freeze.

bsl-py==0.1.11  
astor==0.6.2  
autopep8==1.4  
bleach==1.5.0  
boto==2.48.0  
boto3==1.6.6  
botocore==1.9.6  
bz2file==0.98  
certifi==2018.1.18  
chardet==3.0.4  
Click==7.0  
docutils==0.14  
dominate==2.3.5  
Flask==1.0.2  
Flask-Bootstrap==3.3.7.1  
Flask-WTF==0.14.2  
gast==0.2.0  
gensim==3.4.0  
grpcio==1.10.0  
h5py==2.8.0  
html5lib==0.9999999  
idna==2.6  
itsdangerous==1.1.0  
Jinja2==2.10  
jmespath==0.9.3  
json-lines==0.3.1  
Keras==2.2.2  
Keras-Applications==1.0.4  
Keras-Preprocessing==1.0.2  
Mako==1.0.7  
Markdown==2.6.11  
MarkupSafe==1.0  
nltk==3.2.5  
numpy==1.14.1  
protobuf==3.5.2  
pycodestyle==2.4.0  
python-dateutil==2.6.1  
python-dotenv==0.9.1  
PyYAML==3.12  
requests==2.18.4  
s3transfer==0.1.13  
scikit-learn==0.19.1  
scipy==1.0.0  
six==1.11.0  
sklearn==0.0  
smart-open==1.5.6  
tensorboard==1.6.0  
tensorflow-gpu==1.5.0  
tensorflow-tensorboard==1.5.1  
termcolor==1.1.0  
Theano==1.0.1  
Unidecode==1.0.22  
urllib3==1.22  
visitor==0.1.3  
Werkzeug==0.14.1  
WTForms==2.2.1  

