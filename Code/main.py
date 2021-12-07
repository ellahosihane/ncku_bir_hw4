import os
import  pickle
import argparse
import string
from xml.etree.ElementTree import ProcessingInstruction
import nltk
from nltk.corpus import stopwords
from collections import Counter
import math
import matplotlib.pyplot as plt

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
    parser.add_argument("-f", "--filename", type=str, required=True, help="Filename")
    parser.add_argument("-n", "--sent_num", type=int, required=True, help="Num of sentence per article")
    parser.add_argument("-s", "--sent_weight", action='store_true', help="Show sentence weight")
    parser.add_argument("-w", "--word_weight", action='store_true', help="Show word weight")
    parser.add_argument("-c", "--compare", action='store_true', help="Show the information with same words")
    parser.add_argument("-i", "--important_sent", action='store_true', help="Show the import sentence")
    parser.add_argument("-o", "--output", action='store_true', help="Save result")
    args = parser.parse_args()
    Filename = args.filename
    num = args.sent_num
    Path = f'../Data/{Filename}.pkl'

    with open(Path, 'rb')as fpick:
        document = pickle.load(fpick)
    
    #preprocess
    text_list = loadText(document)
    sentence_list = get_sentence(document)
    countlist = []
    for text in text_list:
        tokens = get_tokens(text)
        filtered = [w for w in tokens if  w not in stopwords.words('english')]
        wo_number = [w for w in filtered if not w.isnumeric()]
        count = Counter(wo_number)
        countlist.append(count)

    word_weight = []
    for i, count in enumerate(countlist):
        weight = {}
        for word in count:
            weight[word] = tfidf(word, count, countlist)
        word_weight.append(weight)

    weight = {}
    for article in word_weight:
        for word, score in article.items():
            if word not in weight.keys():
                weight[word] = 0
            weight[word] += score
    weight = sorted(weight.items(), key=lambda x:x[1], reverse=True)
    word_list = [w[0] for w in weight]
    score_list = [w[1] for w in weight]
    plt.plot(word_list[:10], score_list[:10])
    plt.xticks(rotation=70)
    plt.show()
    if args.word_weight:
        for i, words in enumerate(word_weight):
            print("article", i+1)
            sorted_word = sorted(words.items(), key=lambda x:x[1], reverse=True)
            for w in sorted_word[:num]:
                print(w[0], ':', w[1])
            print()

    sentence_weight = []
    for i, article in enumerate(sentence_list):
        weight = {}
        for j, sentence in enumerate(article):
            words = get_tokens(sentence)
            weight[sentence] = 0
            for w in words:
                if w in word_weight[i].keys():
                    weight[sentence] += word_weight[i][w]
            weight[sentence] /= len(words)
        sentence_weight.append(sorted(weight.items(), key=lambda x:x[1], reverse=True))

    if args.sent_weight:
        for article, title in zip(sentence_weight, document.keys()):
            print("Title: ", title)
            print("Top", num, " sentence")
            for s in article[:num]:
                print(s[0])
            print("")

    if args.compare:
        with open("../Data/same_words.pkl","rb") as fpick:
            same_words = pickle.load(fpick)
        
        print(len(same_words))
        word_set = set()
        for i, words in enumerate(word_weight):
            print("article", i+1)
            sorted_word = sorted(words.items(), key=lambda x:x[1], reverse=True)
            for w in sorted_word[:num]:
                if w[0] in same_words:
                    print(w[0], ':', w[1])
                    word_set.add(w[0])
            print()
        with open(f"../Data/word_set_{Filename}.pkl","wb") as fpick:
            pickle.dump(word_set, fpick)
    
    if args.important_sent:
        with open("../Data/important_words.pkl","rb") as fpick:
            important_words = pickle.load(fpick)
        for article, title in zip(sentence_weight, document.keys()):
            print("Title: ", title)
            print("Top", num, " sentence")
            for s in article[:num]:
                tokens = get_tokens(s[0])
                for w in tokens:
                    if w in important_words:
                        print(s[0])
                        break
            print("")
    if args.output:
        word = []
        for i, words in enumerate(word_weight):
            w = []
            sorted_word = sorted(words.items(), key=lambda x:x[1], reverse=True)
            word.append(sorted_word[:num])
        sents = []
        for article, title in zip(sentence_weight, document.keys()):
            sents.append(article[:num])
        out = {}
        for i, title in enumerate(document.keys()):
            out[title] = (word[i], sents[i])
        
        with open(f"../Data/res_{Filename}.pkl","wb") as fpick:
            pickle.dump(out, fpick)
        
        imp = {}
        with open("../Data/important_words.pkl","rb") as fpick:
            important_words = pickle.load(fpick)
        print(important_words)
        for article, title in zip(sentence_weight, document.keys()):
            for s in article[:num]:
                tokens = get_tokens(s[0])
                for w in tokens:
                    if w in important_words:
                        if title not in imp.keys():
                            imp[title] = []
                        imp[title].append(s[0])
                        break
        with open(f"../Data/imporant_{Filename}.pkl","wb") as fpick:
            pickle.dump(imp, fpick)
        
            
