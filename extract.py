import random
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clear_writing(writing):
    sentence_words = nltk.word_tokenize(writing)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(writing, words):
    sentence_words = clear_writing(writing)
    bag = [0]*len(words)
    for setence in sentence_words:
        for i, word in enumerate(words):
            if word == setence:
                bag[i] = 1
    return(np.array(bag))

def class_prediction(writing, model):
    prevision = bag_of_words(writing, words)
    response_prediction = model.predict(np.array([prevision]))[0]
    results = [[index, response] for index, response in enumerate(response_prediction) if response > 0.25]
    if "1" not in str(prevision) or len(results) == 0 : 
        results = [[0, response_prediction[0]]]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probality": str(r[1])} for r in results]

def get_response(intents, intents_json):
    tag = intents[0]['intent']
    list_of_intents = intents_json['intents']
    for idx in list_of_intents:
        if idx['tag'] == tag:
            result = random.choice(idx['responses'])
            break
    return result


