from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from wtforms.validators import InputRequired

class InputForm(FlaskForm):
  default_claim = "Student accidentally sets college on fire during fireworks proposal."
  default_document = '"He popped the question — and burned down his college sports hall. Hopeless romantic Dim Xiong Chien planned to propose to his girlfriend, Cong Yen, in explosive fashion by setting off fireworks as he got down on one knee. His would-be betrothed didn’t show up but as a last ditch effort, he set off the pyrotechnics in hopes that she’d see the fiery display, according to a report in the Express. The plan bombed, though, when the fireworks set the grass ablaze at the Liaoning Advertisement Vocational College in the city of Shenyang. As firefighters rushed extinguish the massive blaze, Chien, 22, searched for his girlfriend, also 22, who forgot that he had asked her to join him for a walk. “I was feeling a bit surprised that she hadn’t shown up, and was completely unaware that the fireworks had set the grass on fire,” Chien said. “When I found her I said she had to come with me as there was something important I wanted to tell her and show her.”  The fiasco didn’t deter Chien, who decided to postpone the proposal. Yen said she found out later about Chien’s plan but his botched romantic gesture may have cost him the respect of his potential in-laws. “Of course, I love him, but my parents have told me to steer clear, saying he can’t even ask me to marry him without causing a massive hoo-ha,” she said.'
  
  claim = TextAreaField('Claim', [InputRequired()], default=default_claim)
  document = TextAreaField('Document', [InputRequired()], default=default_document)
  submit = SubmitField('Submit')

