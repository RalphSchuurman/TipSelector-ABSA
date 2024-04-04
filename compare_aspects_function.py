import pandas as pd
from icecream import ic
from scipy.stats import fisher_exact

def compare_aspects(hotel_0_summarized,hotel_1_summarized,hotel_2_summarized,hotel_3_summarized,hotel_4_summarized, hotel_5_summarized,significance_level):
    significant_tokens = []
    significant_sentiment = []
    significant_count = []
    p_values = []
    print('significance level is ', significance_level)

    other_hotel_list = ['hotel_1_summarized', 'hotel_2_summarized', 'hotel_3_summarized', 'hotel_4_summarized']
    dfdict = {'Hotel_1': hotel_1_summarized, "Hotel_2": hotel_2_summarized, "Hotel_3": hotel_3_summarized,
              'Hotel_4': hotel_4_summarized, "Hotel_5":hotel_5_summarized}

    for hotel in dfdict.values():
        for i in range(len(hotel_0_summarized.index)):
            lemma_hotel_0 = hotel_0_summarized.at[i, 'Lemma']
            sentiment_hotel_0 = hotel_0_summarized.at[i, 'Sentiment']
            freq_hotel_0 = hotel_0_summarized.loc[i, 'count']
            total_freq_0 = hotel_0_summarized['count'].sum()

            freq_other_hotel = hotel['count'][(hotel['Lemma'] == lemma_hotel_0) & (
                    hotel['Sentiment'] == sentiment_hotel_0)]
            if len(freq_other_hotel) == 0:
                freq_other_hotel = 0
            else:
                freq_other_hotel = int(freq_other_hotel)
            # freq_other_hotel = int(freq_other_hotel)
            total_freq_other = hotel['count'].sum()
            # create table
            contingency_table = pd.DataFrame(0, index=[0, 1], columns=[0, 1])
            # fill the table
            contingency_table.at[0, 0] = freq_hotel_0
            contingency_table.at[0, 1] = freq_other_hotel
            contingency_table.at[1, 0] = (total_freq_0 - freq_hotel_0)
            contingency_table.at[1, 1] = (total_freq_other - freq_other_hotel)
            oddsr, p_value = fisher_exact(contingency_table)
            
            
            
            if p_value < significance_level:
                significant_tokens.append(lemma_hotel_0)
                significant_sentiment.append(sentiment_hotel_0)
                p_values.append(p_value)
                significant_count.append(freq_hotel_0)

    tokens_df = pd.DataFrame(
        {"Token": significant_tokens, 'Sentiment': significant_sentiment, 'Count': significant_count})
    # If there is a positive and negative version of a word significant, delete the one with the lowest count
    tokens_df = tokens_df.sort_values('Count', ascending=True).drop_duplicates(['Token'], keep='last')
    return tokens_df


