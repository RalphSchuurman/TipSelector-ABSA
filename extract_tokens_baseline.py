# Loading packages
import os
import spacy
import re
import pandas as pd
from spacy.symbols import *
import numpy as np
import csv
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import downloader
from nltk.corpus import wordnet
#import nltk.downloader; nltk.download('stopwords')
#nltk.download('sentiwordnet')
from nltk import word_tokenize, pos_tag
from spacy import displacy
from nltk.corpus import stopwords
#nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.tokenize import TweetTokenizer
from icecream import ic


def extract_tokens_baseline(similar_hotels_df,hotel):
    nlp = spacy.load("en_core_web_sm")
    tknzr = TweetTokenizer()

    # hotel_list has the list of the five similar hotels. hotel_of_interest is chosen hotel of interest
    hotel_aspects = get_aspects(hotel)

    return hotel_aspects


def get_aspects(hotel):
    nlp = spacy.load("en_core_web_sm")
    tknzr = TweetTokenizer()
    filename = "./hotel_data/parsed/" + str(hotel)

    print(filename)
    reviews = []
    all_sentences = []
    sentences = []
    sentence_tags = []
    print("Hotel", hotel, "is now being tagged")

    with open(filename, encoding="utf8") as f:
        for line in f:
            x, y = line.strip().split('\t ')
            reviews.append(y.replace('<br/>', ' '))

        for x in reviews:

            sentence = nlp(x).sents
            for z in sentence:
                sentences.append(z)
        token_text = []
        token_pos = []
        token_dep = []
        sentence_number = []
        token_lemma = []
        token_head = []
        token_i = []

        for x in sentences:
            for token in x:
                token_text.append(token.text)
                token_pos.append(token.pos_)
                token_dep.append(token.dep_)
                sentence_number.append(str(x))
                token_i.append(token.i)
                token_head.append(token.head)
                token_lemma.append(token.lemma_)

        sentence_data_df = pd.DataFrame({"Token": token_text, "pos_tag": token_pos, 'dep_tag': token_dep, 'Sentence': sentence_number,
                 'token_lemma': token_i, 'token_head': token_head, 'token_lemma': token_lemma})
            # print(sentence_data_df.head(200).to_string())

    nouns = sentence_data_df.loc[sentence_data_df['pos_tag'] == "NOUN"]
    # print('nouns', nouns.head(200).to_string())

    amod = sentence_data_df.loc[sentence_data_df['dep_tag'] == "amod"]
    amod['Token'] = amod['token_head']
    amod['Token'] = amod['Token'].astype(str)
    amod['Amod_Token'] = amod['token_lemma'].astype(str) + " " + amod['token_head'].astype(str)

    # Delete the singleton nouns that are also in the amod dataframe
    # Token head is equal to noun token, sentence is equal
    # ic(amod['sentence'].iloc[0])
    # ic(nouns['sentence'].iloc[1])

    # ic(amod['Token'].iloc[0])
    # ic(nouns['Token'].iloc[1])

    # common = nouns.merge(amod, on=["Token",'sentence'])
    # print('common',common.head(200).to_string())
    # nouns_without_amod = nouns[(~nouns['Token'].isin(common['Token']))&(~nouns['sentence'].isin(common['sentence']))]
    # Deze werkt niet goed. Verwijderen alleen als token in common zit én sentence in common

    nouns_without_amod = nouns.merge(amod.drop_duplicates(), on=['Token', 'Sentence'],
                                     how='left', indicator=True)
    # print('nouns_without_amod_before drop')
    # print(nouns_without_amod.head(100).to_string())
    nouns_without_amod = nouns_without_amod.loc[nouns_without_amod['_merge'] != 'both']
    nouns_without_amod = nouns_without_amod.drop(
        ['token_head_y', 'pos_tag_y', 'dep_tag_y', 'token_lemma_y', '_merge', 'Amod_Token'], axis=1)
    nouns_without_amod = nouns_without_amod.rename({'pos_tag_x': 'pos_tag', 'dep_tag_x': 'dep_tag'}, axis=1)

    # print('nouns_without_amod_after_rename',nouns_without_amod.head(10).to_string())

    # Change amod token to full amod
    amod['Token'] = amod['Amod_Token']
    # print('amod',amod.head(200).to_string())

    # example compound: capital improvements, air conditioner, metro station
    compounds = sentence_data_df[sentence_data_df['dep_tag'] == 'compound']
    compounds['Token'] = compounds['Token'].astype(str)
    compounds['Compound_token'] = compounds['token_lemma'].astype(str) + " " + compounds['token_head'].astype(str)
    # print('compounds',compounds.head(40).to_string())

    # Remove the compounds from the nouns
    nouns_without_compounds = nouns_without_amod.merge(compounds.drop_duplicates(), on=['Token', 'Sentence'],
                                                       how='left', indicator=True)
    nouns_without_compounds = nouns_without_compounds.loc[nouns_without_compounds['_merge'] != 'both']

    nouns_without_compounds = nouns_without_compounds.drop(
        ['pos_tag_y', 'dep_tag_y', 'token_lemma', 'token_head', '_merge', 'Compound_token'], axis=1)
    nouns_without_compounds = nouns_without_compounds.rename(
        {'pos_tag_x': 'pos_tag', 'dep_tag_x': 'dep_tag', 'token_lemma_x': 'token_lemma', 'token_head_x': 'token_head'},
        axis=1)
    # print('nouns_without_compounds')
    # print(nouns_without_compounds.head(170).to_string()) # capital in 115 should be gone, 80s in 156 should be gone

    # ic(compounds['token_head'].iloc[0]) # is not a string
    # ic(nouns_without_compounds['Token'].iloc[19]) # is a string

    compounds['token_head'] = compounds['token_head'].astype(str)

    nouns_without_compounds_merged = nouns_without_compounds.merge(compounds.drop_duplicates(),
                                                                   left_on=['Token', 'Sentence'],
                                                                   right_on=['token_head', 'Sentence'],
                                                                   how='left', indicator=True)
    nouns_without_compounds_all = nouns_without_compounds_merged.loc[nouns_without_compounds_merged['_merge'] != 'both']
    # print(nouns_without_compounds_all.head(170).to_string()) # improvements in 19 should be gone
    nouns_without_compounds_all = nouns_without_compounds_all.drop(
        ['Token_y', 'pos_tag_y', 'dep_tag_y', 'token_lemma_y', 'token_head_y', 'Compound_token', '_merge'], axis=1)
    nouns_without_compounds_all = nouns_without_compounds_all.rename(
        {'Token_x': 'Aspect', 'pos_tag_x': 'pos_tag', 'dep_tag_x': 'dep_tag', 'token_lemma_x': 'token_lemma',
         'token_head_x': 'token_head'}, axis=1)
    # print('nouns_without_compounds_merged')
    # print(nouns_without_compounds_all.head(170).to_string()) # improvements in 19 should be gone

    # The nouns are in nouns_without_compounds_all
    # The compounds are in compounds
    # The adj modified nouns are in amod

    # Do have to make them ready first to combine (e.g. synsets 1 and 2, tokens for compounds are the complete ones)
    # Make them aspect with list of aspect (e.g. [train, station], Sentiment (NONE) and Sentence

    nouns_without_compounds_all = nouns_without_compounds_all.drop(['pos_tag', 'dep_tag', 'token_lemma', 'token_head'],
                                                                   axis=1)
    nouns_without_compounds_all['Aspect'] = nouns_without_compounds_all['Aspect'].str.split()
    nouns_without_compounds_all['Sentiment'] = ""
    nouns_without_compounds_all = nouns_without_compounds_all[['Aspect', 'Sentiment', 'Sentence']]

    compounds['Aspect'] = compounds['Compound_token'].str.split()
    compounds = compounds.drop(['pos_tag', 'dep_tag', 'token_lemma', 'token_head', 'Token', 'Compound_token'], axis=1)
    compounds['Sentiment'] = ""
    compounds = compounds[['Aspect', 'Sentiment', 'Sentence']]

    amod = amod.drop(['pos_tag', 'dep_tag', 'token_lemma', 'token_head', 'Token'], axis=1)
    amod['Aspect'] = amod['Amod_Token'].str.split()
    amod['Sentiment'] = ""
    amod = amod[['Aspect', 'Sentiment', 'Sentence']]

    # concatenate the three types
    baseline_tokens = pd.concat([nouns_without_compounds_all, compounds, amod], ignore_index = True)
    return(baseline_tokens)