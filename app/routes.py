from flask import render_template, request
from app import app
from app.forms import InputForm 
from run_saved_model import run_model_main

@app.route('/', methods=['GET', 'POST'])
def index():
  form = InputForm(request.form)
  if request.method == 'GET':
    return render_template('input.html', form=form)
  elif request.method == 'POST' and form.validate():
    response = run_model_main(form.claim.data, form.document.data)
    data = []
    for i in range(len(response[0])):
        data.append({
          "num": i,
          "claim": response[0][i],
          "document": response[1][i],
          "related_unrelated_pred": response[2][i],
          "related_unrelated_logit": response[3][i],
          "three_label_pred": response[4][i],
          "three_label_logit": response[5][i]
        })
    return render_template('output.html', data=data)
  else:
    return "OOPS"

