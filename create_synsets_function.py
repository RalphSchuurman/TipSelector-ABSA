## Test file to create synsets

import pandas as pd
import spacy
from nltk.wsd import lesk
from nltk.corpus import wordnet
import nltk
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
from nltk.corpus import wordnet
#from pywsd.lesk import adapted_lesk
#from pywsd.similarity import sim, similarity_by_path
#from pywsd.lesk import simple_lesk
from nltk.stem import WordNetLemmatizer
from pywsd.lesk import adapted_lesk
from nltk.corpus import wordnet as wn
import numpy as np
from icecream import ic
import pickle
import time
import itertools

nlp = spacy.load("en_core_web_sm")
wnl = WordNetLemmatizer()

# noinspection PyTypeChecker
def create_synsets(hotel_aspects):

    hotel_aspects = pd.concat([hotel_aspects, pd.DataFrame(hotel_aspects.Aspect.tolist())], 1)
    hotel_aspects.columns = hotel_aspects.columns.astype(str)

    #print("Hotel_aspects wihtout replace", hotel_aspects.to_string())
    hotel_aspects['Lemma_1'] = hotel_aspects.apply(lambda row: wnl.lemmatize(row['0']), axis=1)
    # If column 1 is !None then also add that to the lemma column. # add space to Lemma_2, to get a space between the two synsets
    hotel_aspects['Lemma_2'] = ["" if str(ele) == "None" else " " + wnl.lemmatize(ele) for ele in hotel_aspects['1']]

    # remove 's in Lemma. So room's becomes room
    #hotel_aspects['Lemma_1'] = hotel_aspects['Lemma_1'].str.replace("'s","")
    #print('hotel_aspects_with_replace',hotel_aspects.to_string())

    #hotel_aspects['Lemma'] = hotel_aspects['Lemma_1'].astype(str) + ' ' + hotel_aspects['Lemma_2']
    hotel_aspects['Lemma'] = hotel_aspects['Lemma_1'].astype(str) + hotel_aspects['Lemma_2']


    # hotel_2_aspects['Synset_1'] = hotel_2_aspects.apply(lambda row: lesk(str(row['Sentence']), str(row['0'])), axis=1)
    hotel_aspects['Synset_1'] = hotel_aspects.apply(lambda row: adapted_lesk(str(row['Sentence']), str(row['0'])),
                                                        axis=1)
    # hotel_2_aspects['Synset_2'] = hotel_2_aspects.apply(lambda row: lesk(str(row['Sentence']), str(row['1'])), axis=1)
    # hotel_2_aspects['Synset_2'] = hotel_2_aspects.apply(lambda row: adapted_lesk(str(row['Sentence']), str(row['1'])), axis=1) adapted lesk works
    # The if function is here not make all the 'None' values the same synset
    hotel_aspects['Synset_2'] = hotel_aspects.apply(
        lambda row: adapted_lesk(str(row['Sentence']), str(row['1'])) if str(row['1']) != 'None' else adapted_lesk(
            str(row['Sentence']), str(row['1']), pos='n'), axis=1)

    hotel_aspects = hotel_aspects.filter(items=['Sentiment', 'Sentence', 'Lemma', 'Synset_1', 'Synset_2'])

    hotel_aspects_df = hotel_aspects

    no_synset = adapted_lesk('The room was clean', 'none', pos='n')  # Create a value for the 'none' synset

    # Replace the 'None' synset values with a Synset('None') value
    hotel_aspects_df['Synset_2'] = hotel_aspects_df.apply(
        lambda row: row['Synset_2'] if str(row['Synset_2']) != 'None' else no_synset, axis=1)
    hotel_aspects_df['Synset_1'] = hotel_aspects_df.apply(
        lambda row: row['Synset_1'] if str(row['Synset_1']) != 'None' else no_synset, axis=1)

    ## make the Synset column a numpy array
    synset_1_numpy = hotel_aspects_df['Synset_1'].to_numpy()
    synset_2_numpy = hotel_aspects_df['Synset_2'].to_numpy()

    #print('with synset', hotel_aspects_df.to_string())

    # Calculate similarity for synset_1 and 2

    print('calculate similarity synset 1 new')

    #word_similarities = dict()

    with open('word_similarity_dict.pickle', 'rb') as openhandle:
        word_similarities = pickle.load(openhandle)

    simi_synset_1 = []
    startnewloop1 = time.time()
    synset_1_combinations = np.array(np.meshgrid(synset_1_numpy, synset_1_numpy)).T.reshape(-1,2)
    for combination in synset_1_combinations:
        key = tuple(sorted([str(combination[0]),str(combination[1])]))
        if key not in word_similarities:
            similarity = wn.synset(combination[0].name()).wup_similarity(wn.synset(combination[1].name()))
            word_similarities[key] = similarity
            simi_synset_1.append(similarity)
        else:
            simi_synset_1.append(word_similarities[key])
    newendloop1 = time.time()

    print('new Loop one took ', newendloop1 - startnewloop1, ' seconds')

    print('calculate similarity synset 2')
    simi_synset_2 = []
    startloop2 = time.time()

    synset_2_combinations = np.array(np.meshgrid(synset_2_numpy, synset_2_numpy)).T.reshape(-1,2)
    for combination in synset_2_combinations:
        key = tuple(sorted([str(combination[0]),str(combination[1])]))
        if key not in word_similarities:
            similarity = wn.synset(combination[0].name()).wup_similarity(wn.synset(combination[1].name()))
            word_similarities[key] = similarity
            simi_synset_2.append(similarity)
        else:
            simi_synset_2.append(word_similarities[key])

    endloop2 = time.time()
    print('New loop two took ', endloop2 - startloop2, ' seconds')

    #with open('word_similarity_dict.pickle', 'wb') as savehandle:
        #pickle.dump(word_similarities, savehandle)
    print('word sim pickle saved')

    # Reshape to use in the where function
    simi_synset_1_array = np.reshape(simi_synset_1, (-1, hotel_aspects_df.shape[0]))
    simi_synset_2_array = np.reshape(simi_synset_2, (-1, hotel_aspects_df.shape[0]))

    # Find which synsets are more than 0.8 percent similar
    result_synset_1 = np.where(simi_synset_1_array > 0.8)
    result_synset_2 = np.where(simi_synset_2_array > 0.8)
    # Create coordinates
    listofCoordinates_synset_1 = list(zip(result_synset_1[0], result_synset_1[1]))
    listofCoordinates_synset_2 = list(zip(result_synset_2[0], result_synset_2[1]))

    # Only keep coordinates that are not equal (because those always have a similarity of 1,0 because it is with themselves)
    print('create coordlist 1')
    startcoordlist1 = time.time()
    coordlist_synset_1 = []
    for cord in listofCoordinates_synset_1:
        if cord[0] != cord[1]:
            coordlist_synset_1.append(cord)
    endcoordlist1 = time.time()
    print('coordlist one took ', endcoordlist1 - startcoordlist1, ' seconds')


    print('create coordlist 2')
    startcoordlist2 = time.time()
    coordlist_synset_2 = []
    for cord in listofCoordinates_synset_2:
        if cord[0] != cord[1]:
            coordlist_synset_2.append(cord)
    endcoordlist2 = time.time()
    print('coordlist two took ', endcoordlist2 - startcoordlist2, ' seconds')
    # Change list into array again
    coordarray_synset_1 = np.array(coordlist_synset_1)
    coordarray_synset_2 = np.array(coordlist_synset_2)

    synset_1_2_combined = np.array(list(set(coordlist_synset_1).intersection(set(coordlist_synset_2))))
    ## Just for testing
    with open('synset_1_2_combined.pkl', 'wb') as f:
        pickle.dump(synset_1_2_combined, f)

    hotel_aspects_df.to_pickle('./hotel_aspects_df')
    ## just for testing
    hotel_aspects_df['Sentence_capital'] = hotel_aspects_df['Sentence']
    hotel_aspects_df['Sentence'] = hotel_aspects_df['Sentence'].str.lower()
    hotel_aspects_df['Lemma'] = hotel_aspects_df['Lemma'].str.lower()
    print('last loop to create ')
    startlastloop = time.time()
    for coord in range(len(synset_1_2_combined)):
        # print(hotel_aspects_df.at[synset_1_2_combined[coord][0],'Synset_1'])
        # print(hotel_aspects_df.at[synset_1_2_combined[coord][0], 'Synset_2'])
        if hotel_aspects_df.at[synset_1_2_combined[coord][0], 'Synset_1'] != no_synset and hotel_aspects_df.at[
            synset_1_2_combined[coord][0], 'Synset_2'] != no_synset and \
                hotel_aspects_df.at[synset_1_2_combined[coord][1], 'Synset_1'] != no_synset and hotel_aspects_df.at[
            synset_1_2_combined[coord][1], 'Synset_2'] != no_synset:
            hotel_aspects_df['Lemma'].iloc[synset_1_2_combined[coord][0]] = hotel_aspects_df['Lemma'].iloc[
                synset_1_2_combined[coord][1]]
    endlastloop = time.time()
    print('last loop took ', endlastloop - startlastloop, ' seconds')

    summarized = hotel_aspects_df.groupby(['Lemma', 'Sentiment']).size().sort_values(ascending=False).reset_index(
        name='count')

    return summarized, hotel_aspects_df

## To do Do not use sentiment if there are very few observations

'''
print('hotel 2')
hotel_2_aspects = pd.read_pickle('hotel_2_aspects.pkl')
print('hotel_2_aspects',hotel_2_aspects.to_string())

hotel_2_summarized, hotel_2_aspects_df = create_synsets(hotel_2_aspects)
print('hotel_2_summarized',hotel_2_summarized)

print('hotel_2_aspects_df', hotel_2_aspects_df.to_string())

#hotel_2_summarized.to_pickle("./hotel_2_summarized.pkl")


print('hotel 0')
hotel_0_aspects = pd.read_pickle('hotel_0_aspects.pkl')
ic(hotel_0_aspects.to_string())
hotel_0_summarized = create_synsets(hotel_0_aspects)
ic(hotel_0_summarized.to_string())
hotel_0_summarized.to_pickle("./hotel_0_summarized.pkl")
'''

'''
    print('calculate similarity synset 1 old')
    simi_synset_1 = []
    startnewloop1 = time.time()
    for x in synset_1_numpy:
        for j in synset_1_numpy:
            key = tuple(sorted([x,j]))

            if key not in word_similarities:
                similarity = wn.synset(x.name()).wup_similarity(wn.synset(j.name()))
                word_similarities[key] = similarity
                simi_synset_1.append(similarity)
            else:
                simi_synset_1.append(word_similarities[key])
    newendloop1 = time.time()
    print('old Loop one took ', newendloop1 - startnewloop1, ' seconds')
'''