import torch
import re
import nltk
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.chat.util import Chat, reflections
from multi_rake import Rake
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk import word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
warnings.filterwarnings(action = 'ignore')   
import gensim 
from gensim.models import Word2Vec
import scipy
'''WordNet links words into semantic relations including synonyms, hyponyms, and meronyms'''
nltk.download('wordnet')
nltk.download('wordnet_ic')
'''Punkt Sentence Tokenizer. This tokenizer divides a 
text into a list of sentences, by using an unsupervised algorithm to build 
a model for abbreviation words, collocations, and words that start sentences'''
nltk.download('punkt')
'''The perceptron part-of-speech tagger implements part-of-speech 
tagging using the averaged, structured perceptron algorithm.'''
nltk.download('averaged_perceptron_tagger')
'''basic stop words'''
nltk.download('stopwords')



messages = []
num_of_msgs = -1

def test():
        score1 = 0
        score2 = 0
        '''Rapid Automatic Keyword Extraction algorithm'''
        rake = Rake()
        thresholdSS = 0.3
        thresholdCD = 0.4
        condition = -1
	#sentence_similarity(messages[num_of_msgs], messages[num_of_msgs-1])
        #cd (messages[num_of_msgs], messages[num_of_msgs-1])
        #cd (messages[num_of_msgs-1], messages[num_of_msgs-2])

        if (len(messages[num_of_msgs]) > 2):
                score1 = cd(messages[num_of_msgs], messages[num_of_msgs-1])
                if (score1):
                        score1 = score1
                else:
                        score1 = 0
        elif (len(messages[num_of_msgs-1]) > 3 and score1 < thresholdCD):
                score2 = cd(messages[num_of_msgs-1], messages[num_of_msgs-2])
                if (score2):
                        score2 = score2
                else:
                        score2 = 0
                        
        if (score1 > thresholdCD and score1 > score2 and len(messages[num_of_msgs-1]) > 3):
                KW = secondary ((messages[num_of_msgs], messages[num_of_msgs-1]))
                if (KW != "Nothing"):
                	print ("Looks like I may have asked for feedback when there wasn't a need for it, if not what could I have said in response to: ", KW)
                else:
                	print ("Looks like I may have asked for feedback when there wasn't a need for it, if not what could I have said")
                
        elif (score2 > thresholdCD and score2 > score1 and len(messages[num_of_msgs-1]) > 3):
                KW = secondary ((messages[num_of_msgs-1], messages[num_of_msgs-2]))
                if (KW != "Nothing"):
                	print ("Looks like I may have asked for feedback when there wasn't a need for it, if not what could I have said in response to: ", KW)
                else:
                	print ("Looks like I may have asked for feedback when there wasn't a need for it, if not what could I have said")	
        else:
                print ("Condition set to 1")
                condition = 1

        if (condition == 1):
                print ("Enter condition1 in Test")
                if (len(messages[num_of_msgs]) > 3):
                        score1 = sentence_similarity(messages[num_of_msgs], messages[num_of_msgs-1])
                        if (score1):
                                score1 = score1
                        else:
                                score1 = 0
                        
                elif (len(messages[num_of_msgs-1]) > 2 and score1 < 0.4):
                        score2 = sentence_similarity(messages[num_of_msgs-1], messages[num_of_msgs-2])
                        if (score2):
                                score2 = score2
                        else:
                                score2 = 0
                        
                if (score1 > thresholdSS and score1 > score2 and len(messages[num_of_msgs-1]) > 3):
                        KW = secondary ((messages[num_of_msgs], messages[num_of_msgs-1]))
                        if (KW != "Nothing"):
                        	print ("Looks like I may have asked for feedback when there wasn't a need for it, if not what could I have said in response to: ", KW)
                        else:
                        	print ("Looks like I may have asked for feedback when there wasn't a need for it, if not what could I have said")	
                        
                elif (score2 > thresholdSS and score2 > score1 and len(messages[num_of_msgs-1]) > 3):
                        KW =secondary ((messages[num_of_msgs-1], messages[num_of_msgs-2]))
                        if (KW != "Nothing"):
                        	print ("Looks like I may have asked for feedback when there wasn't a need for it, if not what could I have said in response to: ", KW)
                        else:
                        	print ("Looks like I may have asked for feedback when there wasn't a need for it, if not what could I have said")	
                else:
                        RE(messages[num_of_msgs])
	

	#keywords = rake.apply(messages[num_of_msgs])

	#print(keywords[:20])

def cd(X, Y):
        if (X == " " and Y == " " and num_of_msgs >= 2):
                X = messages[num_of_msgs]
                Y = messages[num_of_msgs-1]
        elif (X == " " and Y == " "):
                return 2.0
        # tokenization

        if (len(X) >= 3 and len(Y) >= 3):
                X_list = word_tokenize(X)
                Y_list = word_tokenize(Y)
                # sw contains the list of stopwords
                sw = stopwords.words('english')
                l1 =[];l2 =[]
                # remove stop words from string
                X_set = {w for w in X_list if not w in sw}
                Y_set = {w for w in Y_list if not w in sw}
                # form a set containing keywords of both strings
                rvector = X_set.union(Y_set)
                for w in rvector:
                        if w in X_set: l1.append(1) # create a vector
                        else: l1.append(0)
                        if w in Y_set: l2.append(1)
                        else: l2.append(0)
                c = 0
                # cosine formula
                for i in range(len(rvector)):
                        c+= l1[i]*l2[i]
                div = float((sum(l1)*sum(l2))**0.5)
                if (div == 0):
                        return 0.0
                cosine = c / div
                print("Cosine similarity: ", cosine)
                return cosine
        else:
                return 0.0


def RE (message):
        max = 0
        who = re.findall(r'(?:who|whose|whom|whos|who|which\'s)', message)
        if (len(who) > max):
                max = len(who)
                condition = 'who'
        what = re.findall(r'(?:what|whats)', message)
        if (len(what) > max):
                max = len(what)
                condition = 'what'
        when = re.findall(r'(?:when)', message)
        if (len(when) > max):
                max = len(when)
                condition = 'when'
        where = re.findall(r'(?:where)', message)
        if (len(where) > max):
                max = len(where)
                condition = 'where'

        why = re.findall(r'(?:why|whys)', message)
        if (len(why) > max):
                max = len(why)
                condition = 'why'
        if (max == 0):
                KW = secondary (messages[num_of_msgs], messages[num_of_msgs-1])
                if (KW != "Nothing"):
                	print ("RE: Oops! Sorry, What should I have said instead about the topic: ", KW)
                else:
                	KW = secondary (messages[num_of_msgs-1], messages[num_of_msgs-2])
                	if (KW != "Nothing"):
                		print ("RE: Oops! Sorry, What should I have said instead about the topic: ", KW)
                	else:
                		print ("RE: Oops! Sorry, What should I have said instead")	

        else:
                if (condition == 'who'):
                        KW = secondary (messages[num_of_msgs], messages[num_of_msgs-1])
                        print ("Who? Im not really sure, could you tell me more about: ", KW)
                        
                if (condition == 'what'):
                        KW = secondary (messages[num_of_msgs], messages[num_of_msgs-1])
                        print ("Im not really sure about that, could you tell me more about: ", KW)
                        
                if (condition == 'when'):
                        KW = secondary (messages[num_of_msgs], messages[num_of_msgs-1])
                        print ("When? Im not really sure about when exactly, could you tell me more about: ", KW)
                        
                if (condition == 'where'):
                        KW = secondary (messages[num_of_msgs], messages[num_of_msgs-1])
                        print ("Where? Im not really sure where, could you tell me more about: ", KW)
                        
                if (condition == 'why'):
                        KW = secondary (messages[num_of_msgs], messages[num_of_msgs-1])
                        print ("Why? Im not really sure why, could you tell me more about: ", KW)        



            
def secondary(lastmsg,lastmsg1):
        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        lastmsg = pattern.sub('', lastmsg)
        #print (lastmsg)
        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        lastmsg1 = pattern.sub('', lastmsg1)
        #print (lastmsg1)
        rake = Rake()
        rt = 0.0
        rt1 = 0.0
        keywords = rake.apply(lastmsg)
        keywords1 = rake.apply(lastmsg1)

        print ("Rake: Keywords Executed")
        #word = keywords[:1]
        #word = word[0]
        if (keywords):
                word = keywords[:1]
                word = word[0]
                rt = word[1]

        if (keywords1):
                word1 = keywords1[:1]
                word1= word1[0]
                rt1 = word1[1]

        if (keywords and keywords1):
                if (rt1 > rt):
                        return word1[0]
                if (rt1 < rt):
                        return word[0]
                if (rt1 == rt):
                        return word[0]
        else:
                return "Nothing"

'''secondary("Hello my name is a danial", "Hello I wish i was called saim")'''
def secondaryS(lastmsg):
        rake = Rake()
        rt = 0.0
        keywords = rake.apply(lastmsg)
        if (keywords):
                word = keywords[:1]
                word = word[0]
                rt = word[1]
                print (word[0])
                return (word[0])
        else:
                print ("Nothing")
                return "Nothing"


def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def sentence_similarity(sentence1, sentence2):
        """ compute the sentence similarity using Wordnet """
        # Tokenize and tag
        print ("Inside SS")
        #print (sentence1)
        #print (sentence2)       
        sentence1 = pos_tag(word_tokenize(sentence1))
        sentence2 = pos_tag(word_tokenize(sentence2))
        # Get the synsets for the tagged words
        #Synset: a set of synonyms that share a common meaning.
        synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
        synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
        # Filter out the Nones
         # Filter out the Nones
        synsets1 = [ss for ss in synsets1 if ss]
        synsets2 = [ss for ss in synsets2 if ss]
 
        score, count = 0.0, 0
        if (synsets1 and synsets2):
                for syn1 in synsets1:
                        arr_simi_score = []
                        print('=========================================')
                        print(syn1)
                        print('----------------')
                for syn2 in synsets2:
                        print(syn2)
                        simi_score = syn1.path_similarity(syn2)
                        print(simi_score)
                if simi_score is not None:
                        arr_simi_score.append(simi_score)
                        print('----------------')
                        print(arr_simi_score)
                if(len(arr_simi_score) > 0):
                        best = max(arr_simi_score)
                        print(best)
                        score += best
                        count += 1
                # Average the values
                #print('score: ', score)
                #print('count: ', count)
                if (count != 0):
                        score /= count
                        print ("Semantic Analysis Similarity Index")
                        #print (score)
                        return score
                else:
                        return 0
    	

def is_empty(any_structure):
    if any_structure:
        #print('Structure is not empty.')
        return True
    else:
        #print('Structure is empty.')
        return False


def extract (text):
    count = 0
    msgBot = ''
    msgUser = ''
    condition = 0
    for x in text:
        if (count>6 and x != '_' and condition == 0):
            msgBot += x
            count = count + 1
                
        elif (condition == 0):
            if (text[count]=='_' and count > 6):
                condition = 1
                count = 0
            else:
                count = count + 1
            
            
        elif (count >5 and condition==1):
            msgUser += x
            count = count + 1
        else:
            count = count + 1
            
    #print ("extract ftn")
    #print (messages)
    #print (msgBot)
    #print (msgUser)
    global num_of_msgs        
    num_of_msgs = num_of_msgs + 2         
    messages.append(msgBot)
    messages.append(msgUser)
                            
                            
    





"""
pairs = [

[r"i .*(?:said|asked|told).*", ['Sorry I dont know much about food, what could have been an appropriate response?']]
[r"((not|nt|n't).*mak.*sense)|(mak.*no .*sense)", ['hello cat']]

]

chat = Chat(pairs, reflections)
chat.converse("Food")






def sentence_similarity(sentence1, sentence2):
        #compute the sentence similarity using Wordnet 
        # Tokenize and tag
        print ("Inside SS, Print S1, S2")
        print (sentence1)
        print (sentence2)       
        sentence1 = pos_tag(word_tokenize(sentence1))
        sentence2 = pos_tag(word_tokenize(sentence2))
        # Get the synsets for the tagged words
        synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
        synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
        # Filter out the Nones
        
        if (is_empty(synsets1) and is_empty(synsets2)):
                synsets1 = [ss for ss in synsets1 if ss]
                synsets2 = [ss for ss in synsets2 if ss]
                score, count = 0.0, 0
                # For each word in the first sentence
                for synset in synsets1:
                        # Get the similarity value of the most similar word in the other sentence
                        best_score = max([synset.path_similarity(ss) for ss in synsets2])
                        # Check that the similarity could have been computed

                        if best_score is not None:
                                score += best_score
                                count += 1
                                
                if (count == 0):
                        print ("Sentence similarity Score")
                        print (score)
                        return score
                else:
                        score /= count
                        print ("Sentence similarity Score")
                        print (score)
                        return score
        else:
                print ("SS: Oops! Sorry, What should I have said instead?")







	else:
                score = sentence_similarity(messages[num_of_msgs], messages[num_of_msgs-1])
        if (score <= 0.4):
			print ("SCORE< 0.4: Oops! Looks like I changed the topic, What should I have said instead?")
		else:
                        print ("Score > 0.4: Oops! Sorry, What should I have said instead?")

"""

