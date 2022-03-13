import os
from bs4 import BeautifulSoup
import pickle
import pandas as pd
import numpy as np
import spacy
import sys
nlp = spacy.load("en_core_web_sm")
from sklearn import preprocessing
import pickle

# Find the stratum the Hotel of Interest is in
def get_stratum(hotel_of_interest,city):
    owd = os.getcwd()
    print(os.getcwd())
    os.chdir('./hotel_data/landing_pages/' + city)
    filename = hotel_of_interest
    hotel_name_list = []  # Lists all the hotel names for this city

    price_range_list = []  # Lists the price range for each hotel

    hotel_pages = {}  # Dictionary of hotel name : parsed html Tripadvisor page

    if os.path.isfile(filename):
        print("Hotel of interest found in city")
    else:
        print("Error: Hotel of interest not found in city ")

    with open(filename, 'r') as f:
        html_string = f.read()
        hotel_name_list.append(filename)
        hotel_pages[filename] = f.read()

        s2 = 'v:pricerange">'
        ## Price range is not always in the landing page.
        if s2 in html_string:
            price_range = html_string[html_string.index(s2) + len(s2) + 1:html_string.index(s2) + len(s2) + 10]
            price_range = price_range.split("<")
            price_range[0] = price_range[0].split("\n")
            price_range_list.append(price_range[0][0])  # Change it to number of dollar signs
        else:
            sys.exit('No pricerange for this hotel')

    city_list = [city]
    stratum_dict = {"Hotel Name":filename, "Price Range": price_range[0][0], "City":city}
    stratum_df = pd.DataFrame(
        {"Hotel Name": hotel_name_list, "Price Range": price_range_list, "City":city_list})
    print(stratum_dict)
    os.chdir(owd)

    return stratum_dict


# Find hotels in the same stratum
def hotels_same_stratum(city,price):
    owd = os.getcwd()

    # Find the stratum for each hotel
    os.chdir('./hotel_data/landing_pages/' + city)

    hotel_name_list = []  # Lists all the hotel names for this city

    price_range_list = []  # Lists the price range for each hotel
    city_list = []

    hotel_pages = {}  # Dictionary of hotel name : parsed html Tripadvisor page

    number_of_files = len(os.listdir(os.getcwd())) - 1
    c = 1
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".txt"):
            print('Starting file:', filename, "file", c, "of", number_of_files)
            # Load the HTML pages of the hotel on TripAdvisor
            with open(filename, 'r') as f:
                html_string = f.read()
                hotel_name_list.append(filename)
                hotel_pages[filename] = f.read()

            s2 = 'v:pricerange">'
            ## Price range is not always in the landing page.
            if s2 in html_string:
                price_range = html_string[html_string.index(s2) + len(s2) + 1:html_string.index(s2) + len(s2) + 10]
                price_range = price_range.split("<")
                price_range[0] = price_range[0].split("\n")
                price_range_list.append(price_range[0][0])  # Change it to number of dollar signs
            else:
                price_range_list.append("NA")
            city_list.append(city)
            c = c + 1

    stratum_df = pd.DataFrame(
            {"Hotel Name": hotel_name_list, "Price Range": price_range_list, "City": city_list})
    os.chdir(owd)
    return stratum_df

def get_amenities(hotels_same_stratum_list,city,price_stratum):
    print('Getting all the amenities for the chosen city')
    owd = os.getcwd()
    os.chdir('./hotel_data/landing_pages/' + city)
    all_amenities = []  # List the total of amenities that are present in the complete dataset

    hotel_pages = {}  # Dictionary of hotel name : parsed html Tripadvisor page
    hotel_name_list = []  # Lists all the hotel names for this city

    number_of_files = len(hotels_same_stratum_list)
    c = 1
    print("Number of hotels is ", number_of_files)
    amenity_per_hotel = []  # Lists the amenities for each hotel

    for filename in hotels_same_stratum_list:
            print('Amenities: Starting file :', filename, "file", c, "of", number_of_files)
            # Load the HTML pages of the hotel on TripAdvisor
            with open(filename, 'r') as f:
                html_string = f.read()
                hotel_name_list.append(filename)
                hotel_pages[filename] = f.read()

            soup = BeautifulSoup(html_string, 'html.parser')
            # Extracting amenities
            amenity_txt = ''

            amenity_list = []
            for item in soup.select('.amenity_lst'):
                [elem.extract() for elem in soup("span")]
                amenity_txt += item.text
                parts = amenity_txt.split('\n')

                parts = list(filter(None, parts))

                for amenity in parts:
                    all_amenities.append(amenity)
                    amenity_list.append(amenity)
            unique_amenities_set = set(amenity_list)
            unique_amenities_list = list(unique_amenities_set)
            amenity_per_hotel.append(unique_amenities_list)
            c = c + 1
    # Here we create all amenities to do the encoding of the amenities.
    myset = set(all_amenities)
    all_amenities = list(myset)
    print("All amenities in the price stratum set")
    print(all_amenities)
    os.chdir(owd)
    hotel_df = pd.DataFrame({"Hotel Name": hotel_name_list, 'Amenities': amenity_per_hotel})
    with open('./all_amenities_' + city + price_stratum +  '.pkl', 'wb') as u:
        pickle.dump(all_amenities, u)
    # Do one hot encoding with amenities of a hotel and all amenities
    lb = preprocessing.MultiLabelBinarizer()
    lb.fit([all_amenities])
    print(lb.classes_)
    hotel_mlb = lb.transform(hotel_df['Amenities'])
    print(hotel_mlb)
    hotel_mlb_df = pd.DataFrame(hotel_mlb)
    print(hotel_mlb_df)
    amenity_df = pd.concat([hotel_df, hotel_mlb_df], axis=1)
    print(amenity_df)

    return amenity_df

def get_hotel_name(city, hotel_number):
    #os.chdir('./hotel_data/landing_pages/' + city)
    hotel_name_descr_list = []
    for filename in os.listdir('./hotel_data/landing_pages/' + city):
        if filename.endswith(".txt") and filename in hotel_number :
            print('Getting hotel name of:', filename)
            # Load the HTML pages of the hotel on TripAdvisor
            with open('./hotel_data/landing_pages/' + city + '/' + filename, 'r') as f:
                html_string = f.read()

            soup = BeautifulSoup(html_string, 'html.parser')

            # Name of the hotel
            s3 = '"warLocName">'
            if s3 in html_string:
                hotel_name_descr = html_string[html_string.index(s3) + len(s3) + 0:html_string.index(s3) + len(s3) + 30]
                hotel_name_descr = hotel_name_descr.split("<")
                hotel_name_descr[0] = hotel_name_descr[0].split("\n")
                hotel_name_descr_list.append(hotel_name_descr[0][0])  # Change it to number of dollar signs
            else:
                hotel_name_descr_list.append("NA")

    return(hotel_name_descr_list)

