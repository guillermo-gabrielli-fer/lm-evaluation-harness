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
 
def doc_to_choice(doc) -> str:
   return [choice['text'] for choice in doc['answers']]
   
def doc_to_target(doc) -> int:
   """ Returns the index of the correct answer (4 or 5 options)"""
   return ['А', 'Б', 'В', 'Г', 'Д'].index(doc['correct_answers'][0])

def doc_to_text(doc) -> str:
    """ Returns the text of the question"""
    choices = "\n".join([f"{choice['marker']}. {choice['text']}" for choice in doc['answers']])
    return f"{doc['question'].strip()}\n{choices}\nВідповідь:"
