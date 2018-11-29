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
    data = parse_response(response)
    return render_template('output.html', data=data)
  else:
    return "OOPS"


def parse_response(response):
  data = {}

  data['overall_data'] = {
    "claim": response[0][0],
    "document": response[1][0],
    "related_unrelated_pred": response[2][0],
    "related_unrelated_logit": round_decimals(response[3][0]),
    "related_unrelated_strengths": compute_strengths(response[3][0]),
    "three_label_pred": response[4][0],
    "three_label_logit": round_decimals(response[5][0]),
    "three_label_strengths": compute_strengths(response[5][0])
  }

  data['sentence_data'] = []
  for i in range(1, len(response[0])):
      data['sentence_data'].append({
        "num": i,
        "claim": response[0][i],
        "sentence": response[1][i],
        "related_unrelated_pred": response[2][i],
        "related_unrelated_logit": round_decimals(response[3][i]),
        "related_unrelated_strengths": compute_strengths(response[3][i]),
        "three_label_pred": response[4][i],
        "three_label_logit": round_decimals(response[5][i]),
        "three_label_strengths": compute_strengths(response[5][i])
      })

  return data;

def round_decimals(logits):
    return [round(logit, 2) for logit in logits]

def compute_strengths(logits):
  # 0: [0,0.2) 1: [0.2,0.4) 2: [0.4,0.6) 3: [0.6,0.8) 4: [0.8,1.0] 
  strengths = []
  for logit in logits:
    strengths.append(int(logit//0.2))
  return strengths
      

