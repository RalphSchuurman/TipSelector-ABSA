import pandas as pd



def get_sentences(significant_tokens, hotel_0_aspects_df):
    hotel_0_aspects_df = hotel_0_aspects_df.filter(items=['Sentiment', 'Lemma', 'Sentence_capital'])
    ## Delete sentences that do not contain any of the significant tokens
    # Join sentiment and tokens
    significant_tokens['TokenSentiment'] = significant_tokens['Token'] + significant_tokens['Sentiment']
    hotel_0_aspects_df['TokenSentiment'] = hotel_0_aspects_df['Lemma'] + hotel_0_aspects_df['Sentiment']

    token_list = significant_tokens['TokenSentiment'].tolist()
    token_set = set(token_list)

    # Set generic terms to delete them. Maybe better to create these by looking at what aspect words are used most in all review
    generic_terms = ['hotelNEG', 'hotelNEU', 'hotelPOS', 'placeNEG', 'placeNEU', 'placePOS', 'hostelNEG', 'hostelNEU',
                     'hostelPOS']
    generic_terms_set = set(generic_terms)

    token_set = token_set.difference(generic_terms_set)
    token_set_short = token_set
    token_set_shortest = token_set
    optimal_sentences = []
    sentence_tokens = []

    while len(token_set) != 0:
        token_list_2 = list(token_set)
        mask = hotel_0_aspects_df['TokenSentiment'].isin(token_list_2)
        sentences_with_significant_tokens = hotel_0_aspects_df[mask]
        df1 = sentences_with_significant_tokens.groupby('Sentence_capital')['TokenSentiment'].apply(list).reset_index(
            name='tokens')
        df1['len'] = df1['tokens'].str.len()
        df1['sentence_length'] = df1['Sentence_capital'].str.len()
        # df1 = df1.sort_values(by='len', ascending=False, ignore_index=True).drop(columns='len') only sort on # tokens, not length of the sentence
        df1 = df1.sort_values(by=['len', 'sentence_length'], ascending=[False, False], ignore_index=True).drop(
            columns=['len', 'sentence_length'])
        optimal_sentences.append(df1.at[0, 'Sentence_capital'])
        sentence_tokens.append(df1.at[0, 'tokens'])
        token_row_set = set(df1.at[0, 'tokens'])
        token_set = token_set.difference(token_row_set)

    sentences_df = pd.DataFrame({"Tokens": sentence_tokens, "Sentence": optimal_sentences}) # df to see tokens for each sentence

    optimal_sentences_short = []
    sentence_tokens_short = []

    while len(token_set_short) != 0:
        token_list_2 = list(token_set_short)
        mask = hotel_0_aspects_df['TokenSentiment'].isin(token_list_2)
        sentences_with_significant_tokens = hotel_0_aspects_df[mask]
        df1 = sentences_with_significant_tokens.groupby('Sentence_capital')['TokenSentiment'].apply(list).reset_index(
            name='tokens')
        df1['len'] = df1['tokens'].str.len()
        df1['sentence_length'] = df1['Sentence_capital'].str.len()
        df1 = df1.sort_values(by=['len', 'sentence_length'], ascending=[False, True], ignore_index=True).drop(
            columns=['len', 'sentence_length']) #When you want to sort for tokens covered first, then sentence length
        optimal_sentences_short.append(df1.at[0, 'Sentence_capital'])
        sentence_tokens_short.append(df1.at[0, 'tokens'])
        token_row_set = set(df1.at[0, 'tokens'])
        token_set_short = token_set_short.difference(token_row_set)

    sentences_df_short = pd.DataFrame({"Tokens": sentence_tokens_short, "Sentence": optimal_sentences_short})

    optimal_sentences_shortest = []
    sentence_tokens_shortest = []

    while len(token_set_shortest) != 0:
        token_list_2 = list(token_set_shortest)
        mask = hotel_0_aspects_df['TokenSentiment'].isin(token_list_2)
        sentences_with_significant_tokens = hotel_0_aspects_df[mask]
        df1 = sentences_with_significant_tokens.groupby('Sentence_capital')['TokenSentiment'].apply(list).reset_index(
            name='tokens')
        df1['sentence_length'] = df1['Sentence_capital'].str.len()
        # df1 = df1.sort_values(by='len', ascending=False, ignore_index=True).drop(columns='len')
        df1 = df1.sort_values(by='sentence_length', ascending=True, ignore_index=True).drop(columns='sentence_length') # Only choose based on sentence length, not # tokens covered
        #df1 = df1.sort_values(by=['len', 'sentence_length'], ascending=[False, True], ignore_index=True).drop(
           # columns=['len', 'sentence_length']) When you want to sort for tokens covered first, then sentence length
        optimal_sentences_shortest.append(df1.at[0, 'Sentence_capital'])
        sentence_tokens_shortest.append(df1.at[0, 'tokens'])
        token_row_set = set(df1.at[0, 'tokens'])
        token_set_shortest = token_set_shortest.difference(token_row_set)

    sentences_df_shortest = pd.DataFrame({"Tokens": sentence_tokens_shortest, "Sentence": optimal_sentences_shortest})

    return sentences_df, sentences_df_short, sentences_df_shortest

