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
machine name:port
```   

Normally the port number is 5000 by default. So, if you run it on the beorn machine it will be 

```   
beorn.csail.mit.edu:5000
```   

Note that some SLS machines don't allow remote connections, such as sls-quad-30, so use a different one.


