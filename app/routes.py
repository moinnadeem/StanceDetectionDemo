from flask import render_template, request
from app import app
from app.forms import InputForm 

@app.route('/', methods=['GET', 'POST'])
def index():
  form = InputForm(request.form)
  if request.method == 'GET':
    return render_template('input.html', form=form)
  elif request.method == 'POST' and form.validate():
    data = {'claim': form.claim.data,
            'document': form.document.data}
    return render_template('output.html', data=data)
  else:
    return "OOPS"

