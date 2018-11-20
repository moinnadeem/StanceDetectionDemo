from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import InputRequired

class InputForm(FlaskForm):
  claim = StringField('Claim', [InputRequired()])
  document = StringField('Document', [InputRequired()])
  submit = SubmitField('Submit')

