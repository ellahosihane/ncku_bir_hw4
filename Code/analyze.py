import os
import  pickle
import argparse
import string
from xml.etree.ElementTree import ProcessingInstruction
import nltk
from nltk.corpus import stopwords
from collections import Counter


def get_tokens(text):
    lowers = text.lower()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lowers.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

def get_sentence(document):
    sentence_list = []
    for _, article in document.items():
        doc = []
        for _, content in article.items():
            if type(content) == type(""):
                doc.extend(nltk.sent_tokenize(content))
        sentence_list.append(doc)
    return sentence_list

def loadText(document):
    text_list = []
    for _, article in document.items():
        doc = ""
        for _, content in article.items():
            if type(content) == type(""):
                doc += content
        text_list.append(doc)
    return text_list 

def tf(word, count):
    return count[word] / sum(count.values())
def idf(word, count_list):
    num_doc = sum(1 for count in count_list if word in count)
    return math.log(len(count_list) / num_doc)
def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", "--filename1", type=str, required=True, help="Filename1")
    parser.add_argument("-f2", "--filename2", type=str, required=True, help="Filename2")
    parser.add_argument("-m", "--mode", type=str, required=True, choices=['Same_word', 'Tfidf'], help="Mode")

    args = parser.parse_args()
    File1 = args.filename1
    File2 = args.filename2

    Path1 = f'../Data/{File1}.pkl'
    with open(Path1, 'rb')as fpick:
        document1 = pickle.load(fpick)
    
    Path2 = f'../Data/{File2}.pkl'
    with open(Path2, 'rb')as fpick:
        document2 = pickle.load(fpick)

    if args.mode == 'Same_word':
        text_list1 = loadText(document1)
        text_list2 = loadText(document2)
        words1 = []
        words2 = []
        for text in text_list1:
            tokens = get_tokens(text)
            filtered = [w for w in tokens if  w not in stopwords.words('english')]
            wo_number = [w for w in filtered if not w.isnumeric()]
            words1.extend(wo_number)
        for text in text_list2:
            tokens = get_tokens(text)
            filtered = [w for w in tokens if  w not in stopwords.words('english')]
            wo_number = [w for w in filtered if not w.isnumeric()]
            words2.extend(wo_number)
        same_words = set(words1) & set(words2)
        
        with open("../Data/same_words.pkl","wb") as fpick:
            pickle.dump(same_words, fpick)
    else:
        same_words = document1 & document2
        print(len(same_words))
        with open("../Data/important_words.pkl","wb") as fpick:
            pickle.dump(same_words, fpick)

    