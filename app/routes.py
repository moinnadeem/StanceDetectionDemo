from flask import render_template
from app import app
from app.forms import InputForm 

@app.route('/')
@app.route('/index')
def index():
  form = InputForm() 
  return render_template('input.html', form=form)

