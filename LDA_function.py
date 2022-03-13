import os
from bs4 import BeautifulSoup
import pickle
import pandas as pd
import numpy as np
import spacy
import scipy.sparse
nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction.text import CountVectorizer
from gensim import matutils, models
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import gensim.corpora as corpora
import pickle

#create LDA for one city

def LDA_hotels(hotels_same_stratum_list,city,price_stratum):
    import os
    from bs4 import BeautifulSoup
    import pickle
    import pandas as pd
    import numpy as np
    import spacy
    import scipy.sparse
    nlp = spacy.load("en_core_web_sm")
    from sklearn.feature_extraction.text import CountVectorizer
    from gensim import matutils, models
    import gensim
    from gensim.utils import simple_preprocess
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    import re
    import gensim.corpora as corpora
    import pickle
    owd = os.getcwd()
    os.chdir('./hotel_data/parsed/')


    hotel_reviews = {}  # Dictionary of hotel name : reviews

    number_of_files = len(hotels_same_stratum_list)
    c = 1
    print("Number of hotels is ", number_of_files)

    for filename in hotels_same_stratum_list:
            print('Opening reviews for LDA from file :', filename, "file", c, "of", number_of_files)
            # Open reviews
            sentences = []
            review_filename = filename
            reviews = []
            with open(review_filename) as g:
                for line in g:
                    x, y = line.strip().split('\t ')
                    y = y.lower()
                    reviews.append(y.replace('<br/>', ' '))
                total_review = []
                total_review.append(reviews)
                hotel_reviews[filename] = total_review
            c = c + 1

    pdhotel = pd.DataFrame.from_dict(hotel_reviews, orient='index')
    pdhotel.reset_index(level=0, inplace=True)
    pdhotel.columns = ['hotel', 'text']

    pdhotel['text'] = [','.join(map(str, l)) for l in pdhotel['text']]
    pdhotel['text'] = pdhotel['text'].map(lambda x: re.sub('[,\.!?-]', '', x))

    from nltk.tokenize import RegexpTokenizer
    from nltk.stem.porter import PorterStemmer
    from gensim import corpora, models
    import gensim

    tokenizer = RegexpTokenizer(r'\w+')
    ## LDA Code fromh ttps://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
    # load stop words
    stop_words = stopwords.words('english')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # list for tokenized documents in loop
    texts = []

    doc_set = pdhotel.text.values.tolist()
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in stop_words]

        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        # add tokens to list
        texts.append(stemmed_tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    # Set amount of topics
    topics = 40

    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topics, id2word=dictionary, passes=2)
    get_document_topics = ldamodel.get_document_topics(corpus, minimum_probability=0, minimum_phi_value=None,
                                                       per_word_topics=False)
    topic_probability = []
    for i in range(len(corpus)):
        prob_per_topic = []
        for j in range(len(get_document_topics[0])):
            prob_per_topic.append(get_document_topics[i][j][1])
        topic_probability.append(prob_per_topic)

    topic_df = pd.DataFrame(topic_probability)
    LDA_df = pd.concat([pdhotel, topic_df], axis=1)
    LDA_df = LDA_df.drop('text', 1)
    print("LDA done")
    os.chdir(owd)
    return LDA_df