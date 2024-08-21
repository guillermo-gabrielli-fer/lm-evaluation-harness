import datasets

def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
      # modifies the contents of a single
      # document in our dataset.
      doc["choices"] = [doc["choice1"], doc["choice2"], doc["wrong_answer"]]
      doc["gold"] = doc["label"]
      return doc

    return dataset.map(_helper) # returns back a datasets.Dataset object
