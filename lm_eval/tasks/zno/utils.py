import datasets

def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
      # modifies the contents of a single
      # document in our dataset.
      doc["choices"] = [doc["choice1"], doc["choice2"], doc["wrong_answer"]]
      doc["gold"] = doc["label"]
      return doc

    return dataset.map(_helper) # returns back a datasets.Dataset object



"""
{
    "question": "Кого з великих київських князів історик О. Субтельний охарактеризував\nтак:\n«Його слов’янське ім’я, варязьке виховання, кочовий спосіб життя віддзеркалювали поєднання європейського та азіатського начал. Його управління\nознаменувало апогей ранньої героїчної доби в історії Київської Русі»?",
    "answers": [
        {
            "marker": "А",
            "text": "Олега"
        },
        {
            "marker": "Б",
            "text": "Ігоря"
        },
        {
            "marker": "В",
            "text": "Святослава"
        },
        {
            "marker": "Г",
            "text": "Володимира"
        }
    ],
    "correct_answers": [
        "В"
    ],
    "subject": "ukrainian-history"
}
"""
 
MODE = 'marker' # 'marker', 'text', or 'both'

def doc_to_choice(doc,mode='marker') -> str:
    """ 
    Returns the list of choices for the given document 
    This is what lm-eval will measure the likelihood of.
    Args:
    doc: the document to extract the choices from
    mode: how to format the choice to compute its likelihood
        - 'marker': only the marker (Ukrainian letters)
        - 'text': only the text of the choice
        - 'both': both the marker and the text, separated by a dot
    """
    if mode == 'marker':
       return [f"{choice['marker']}" for choice in doc['answers']]
    elif mode == 'text':
       return [f"{choice['text']}" for choice in doc['answers']]
    elif mode == 'both':
       return [f"{choice['marker']}. {choice['text']}" for choice in doc['answers']]
    else: 
        raise ValueError(f"Invalid mode: {mode}")
   
def doc_to_target(doc) -> int:
   """ Returns the index of the correct answer (4 or 5 options)"""
   return ['А', 'Б', 'В', 'Г', 'Д'].index(doc['correct_answers'][0])

def doc_to_text(doc,mode='marker',answer=True) -> str:
    """ 
    Returns the prompt to be sent to the model 
    We will send the same prompt regardless of what we are measuring the 
    likelihood of, we change it in the choice field instead.
    """
    choices = "\n".join([f"{choice['marker']}. {choice['text']}" for choice in doc['answers']])
    suffix = "\nТвоя відповідь повинна містити тільки букву." if mode == 'marker' else ''
    suffix += "\nВідповідь:" if answer else ""
    return f"Ти повинен вибрати єдиний правильний варіант. \n{doc['question'].strip()}\n{choices}{suffix}"

