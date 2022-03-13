from amenity_function import get_amenities
from amenity_function import get_stratum, hotels_same_stratum
from extract_tokens_baseline import extract_tokens_baseline
from create_synsets_function import create_synsets
from compare_aspects_function import compare_aspects
from amenity_function import get_hotel_name
from cover_tokens_function import get_sentences
from LDA_function import LDA_hotels
import pandas as pd
import pickle
from icecream import ic
import os
import sys
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path

# City dictionary
city_dict = {"Washington_DC":"g28970","Phoenix":"g31310","Los_Angeles":"g32655",
             "Orlando":"g345151" , "Chicago":"g35805","Las_Vegas":"g45963","Dallas":"g55711",
                "Houston":"g56003","San_Francisco":"g60713","New_York":"g60763","Philadelphia":"g60795",
"Atlanta":"60898","San_Antonio":"60956","London":"g186338","Paris":"187147","Berlin":"187323","Barcelona":"187497"}

user_input_city = "Washington_DC"
hotel_of_interest = "d84033.txt"

city = city_dict[user_input_city]
owd = os.getcwd()

#Get stratum of the hotel of interest
stratum_dict = get_stratum(hotel_of_interest,city)
price_stratum = stratum_dict["Price Range"]

# Find hotels in the same stratum
hotels_same_stratum_df = hotels_same_stratum(city,price_stratum)

print('Remove hotels that are not in stratum')
hotels_same_stratum_df = hotels_same_stratum_df[hotels_same_stratum_df['Price Range'] == price_stratum]
print(hotels_same_stratum_df) # dataframe with hotels in same stratum
# Change to list
hotels_same_stratum_list = hotels_same_stratum_df['Hotel Name'].to_list()

# Create the amenitiy vector for the hotels in the same stratum
amenity_df = get_amenities(hotels_same_stratum_list,city,price_stratum)
# Drop the column that describes the amenities
amenity_df = amenity_df.drop('Amenities',1)

## Perform LDA for the hotels in the same list
LDA_df = LDA_hotels(hotels_same_stratum_list,city,price_stratum)


LDA_df = LDA_df.add_prefix("LDA_") # Put LDA in front of the variables

# Create similarity DF, which consists of the amenities and the LDA value
similarity_df = amenity_df.set_index('Hotel Name').join(LDA_df.set_index('LDA_hotel'))

# From this similarity, get for each hotel its 5 most similar

distance_matrix = cdist(similarity_df, similarity_df, 'euclid')
print(distance_matrix)

top_5_hotels = np.argsort(distance_matrix, axis=1)[:,1:6]  # Only show top 5 hotels, ignoring the first one
top_5_hotels_df = pd.DataFrame(top_5_hotels, index=similarity_df.index.copy())
top_5_hotels_df.reset_index(level=0, inplace=True)
top_5_hotels_df.columns = ['HotelName', 'First', 'Second', 'Third', 'Fourth', 'Fifth']
dictionary = pd.Series(top_5_hotels_df.HotelName.values, index=top_5_hotels_df.index).to_dict()
similar_hotels_df = top_5_hotels_df.drop('HotelName', axis=1).stack().map(dictionary).unstack().set_index(
        similarity_df.index.copy())
similar_hotels_df = similar_hotels_df.reset_index()
all_hotel_names = similar_hotels_df['Hotel Name'].tolist()


hotels = similar_hotels_df.loc[similar_hotels_df["Hotel Name"] == hotel_of_interest] # Get the similar hotels of the hotel of interest
hotel_list = hotels.values.flatten().tolist() # Make it a list
print(hotel_list)

# Check if hotel of interest does not have emptry reviews
print('size of review is ', os.path.getsize('./hotel_data/parsed/' + hotel_of_interest))
print('Creating tips for ', hotel_of_interest)
if os.path.getsize('./hotel_data/parsed/' + hotel_of_interest) == 0:
    print('hotel ', hotel_of_interest, ' has no reviews and thus will be skipped')
    sys.exit("The hotel of interest has no reviews and thus will be skipped")

hotel_aspects_baseline = {}
# loading hotel aspects with the baseline

for hotel in hotel_list:
    if os.path.isfile('./baseline_temp_data/' + hotel + '_aspects_baseline.pkl'):
        print(hotel + " is found in folder")
        hotel_aspects_baseline[hotel] = pd.read_pickle('./baseline_temp_data/' + hotel + '_aspects_baseline.pkl')
    else:
        print('extracting hotel ' + hotel + ' tokens')

        if os.path.getsize('./hotel_data/parsed/' + hotel) == 0:
            print('hotel ', hotel, ' is not of interest but has no reviews and thus will be skipped')
            continue;

        hotel_aspects_baseline[hotel] = extract_tokens_baseline(similar_hotels_df, hotel)
        hotel_aspects_baseline[hotel].to_pickle('./baseline_temp_data/' + hotel + '_aspects_baseline.pkl')

hotel_aspects_df_dict_baseline = {}
hotel_summarized_dict_baseline = {}

for hotel in hotel_list:
    if os.path.getsize('./hotel_data/parsed/' + hotel) == 0:
        print('hotel ', hotel, ' has no reviews and thus will be skipped')
        continue;

    if os.path.isfile('./baseline_temp_data/' + hotel + '_summarized_baseline.pkl'):
        print(hotel + " summarized and aspects_df is found in folder")
        hotel_aspects_df_dict_baseline[hotel] = pd.read_pickle(
            './baseline_temp_data/' + hotel + '_aspects_df_baseline.pkl')
        hotel_summarized_dict_baseline[hotel] = pd.read_pickle(
            './baseline_temp_data/' + hotel + '_summarized_baseline.pkl')
    else:
        print('creating synsets hotel ' + hotel + ' tokens')
        hotel_summarized_dict_baseline[hotel], hotel_aspects_df_dict_baseline[hotel] = create_synsets(
            hotel_aspects_baseline[hotel])
        hotel_summarized_dict_baseline[hotel].to_pickle(
            './baseline_temp_data/' + hotel + '_summarized_baseline.pkl')
        hotel_aspects_df_dict_baseline[hotel].to_pickle(
            './baseline_temp_data/' + hotel + '_aspects_df_baseline.pkl')

# Set significance level for Fisher test above
significance_levels = np.array([0.05, 0.01])

# Account for some reviews that may be empty

for i in range(1,6):
    if os.path.getsize('./hotel_data/parsed/' + hotel_list[i]) == 0:
        print("hotel dict " + hotel_list[i] + " is empty and will be created from others")
        if i == 5:
            hotel_summarized_dict_baseline[hotel_list[i]] = hotel_summarized_dict_baseline[hotel_list[i - 1]]
        else:
            hotel_summarized_dict_baseline[hotel_list[i]] = hotel_summarized_dict_baseline[hotel_list[i + 1]]

for significance_level in significance_levels:
    if os.path.isfile('./baseline_temp_data/' + hotel_list[0] + "_" + str(significance_level) + '_significant_tokens_baseline.pkl'):
        print(hotel_list[0] + ' significant tokens found in folder')
        significant_tokens_baseline = pd.read_pickle(
            './baseline_temp_data/' + hotel_list[0] + "_" + str(significance_level) + '_significant_tokens_baseline.pkl')
    else:
        print(hotel_list[0] + ' significant tokens not found in folder, will be created')
        significant_tokens_baseline = compare_aspects(hotel_summarized_dict_baseline[hotel_list[0]],
                                                      hotel_summarized_dict_baseline[hotel_list[1]],
                                                      hotel_summarized_dict_baseline[hotel_list[2]],
                                                      hotel_summarized_dict_baseline[hotel_list[3]],
                                                      hotel_summarized_dict_baseline[hotel_list[4]],
                                                      hotel_summarized_dict_baseline[hotel_list[5]], significance_level)
        significant_tokens_baseline.to_pickle('./baseline_temp_data/' + hotel_list[0] + "_" + str(significance_level) + '_significant_tokens_baseline.pkl')

    if os.path.isfile("./Output_sentences_baseline/Shortest/" + str(significance_level) + '/' + hotel_list[0] + "_" + str(significance_level) + "_sentence_df_shortest_baseline.xlsx"):
        print(hotel_list[0] + ' is already summarized in sentences')

    else:
        print(hotel_list[0] + ' is not summarized in sentences. Will be created.')
        # Get the hotel name to remove from the tokens
        hotel_name = get_hotel_name(city, hotel_of_interest)
        hotel_name_split = ' '.join(hotel_name).split()
        hotel_name_split = list(map(str.lower, hotel_name_split))
        # Remove hotel name from the tokens
        mask = significant_tokens_baseline['Token'].str.contains(r'\b(?:{})\b'.format('|'.join(hotel_name_split)))
        significant_tokens_baseline = significant_tokens_baseline[~mask]
        ## Change directory back to the original one
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        # transform the aspects to select the sentences
        sentences_df_baseline, sentences_df_short_baseline, sentence_df_shortest_baseline = get_sentences(
            significant_tokens_baseline, hotel_aspects_df_dict_baseline[hotel_list[0]])



        Normal_path = Path('./Output_sentences_baseline/Standard/'+ str(significance_level)+'/')
        Normal_path.mkdir(parents=True, exist_ok=True)
        Short_path = Path('./Output_sentences_baseline/Hybrid/'+ str(significance_level)+'/')
        Short_path.mkdir(parents=True, exist_ok=True)
        Shortest_path = Path('./Output_sentences_baseline/Short/'+ str(significance_level)+'/')
        Shortest_path.mkdir(parents=True, exist_ok=True)

        sentences_df_baseline.to_excel(
            "./Output_sentences_baseline/Standard/" + str(significance_level) + '/' + hotel_list[0] + "_" + str(
                significance_level) + "_sentences_df_baseline.xlsx")
        sentences_df_short_baseline.to_excel(
            "./Output_sentences_baseline/Hybrid/" + str(significance_level) + '/' + hotel_list[0] + "_" + str(
                significance_level) + "_sentences_df_hybrid_baseline.xlsx")
        sentence_df_shortest_baseline.to_excel(
            "./Output_sentences_baseline/Short/" + str(significance_level) + '/' + hotel_list[0] + "_" + str(
                significance_level) + "_sentence_df_short_baseline.xlsx")

        print('Done with creating tips for', hotel_of_interest)
        print('The sentences are written to ./Output_sentence_baseline/')