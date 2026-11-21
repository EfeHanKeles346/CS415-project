from pathlib import Path
import json
path=Path('tmp/proposal/build_proposal.py')
text=path.read_text()
paragraphs=[
'People often write reviews after buying products online. They share their opinions about the product and give it a rating from one to five stars. In this project, we want to build a deep learning model that predicts this rating by reading the review. The model will use only the review text to make its prediction. We will use the Amazon Fine Food Reviews dataset, which includes food reviews and their star ratings [9].',
'During training, the model will learn from reviews and their correct ratings. Later, we will give it reviews that it has not seen before. It will predict a rating for each review. We will then compare these predictions with the real ratings to check how well the model works. The model will choose one of five classes: one, two, three, four or five stars.',
'This task can be difficult because one review may include both positive and negative opinions. For example, a customer may write, "The taste is good, but the price is too high." The customer may give this product three or four stars. Different people can also give different ratings for similar experiences. We therefore need to measure both how often the model is correct and how far its wrong predictions are from the real ratings. Predicting two stars instead of one is a smaller error than predicting five stars instead of one [1].',
'A model like this could help companies understand customer feedback when a written review has no star rating. We plan to use a convolutional neural network, or CNN, to learn useful patterns in groups of words [2]. We also want to test the model on reviews of products that were not included in its training data. This will show how well it works on new food products. However, good results on food reviews do not mean that the model will also work well on electronics reviews or reviews in another language. Testing on another product category would be a separate step [7].'
]
start=text.index('problem=[')
end=text.index('\nfor text in problem:',start)
text=text[:start]+'problem='+repr(paragraphs)+text[end:]
path.write_text(text)
Path('tmp/proposal/problem_b1.json').write_text(json.dumps(paragraphs,indent=2))
print('Problem definition words:',sum(len(p.split()) for p in paragraphs))
