# Aspect-Based Sentiment Analysis for Tip Mining
##### Ralph Schuurman, Flavius Frasincar & Jasmijn Klinkhamer

This repo is for an addition of TipSelector by Zhu et al. (2018). It adds Aspect-Based Sentiment Analysis (ABSA) to the method and the possibility to create smaller tips. In the repo my interpretation of the method from Zhu et al. (2018) can also be found. The implementation is done in Python 3.8. The post-trained BERT_Review model from Xu et al. (2019) was used in combination with the ABSA layer from Li. et al (2019).

## How to use

- Install the required libraries from the requirements.txt file
- Paste the two TipSelector data files (landing_pages and parsed) in the hotel_data file. These can be found on https://tinyurl.com/TipSelectorData and was prepared by Zhu et al. (2018).
- Download the BERT model from this GDrive link and unzip: https://drive.google.com/file/d/1x_GIdWwMuLZvnuoXAgStt8w4JyAX-2sN/view?usp=sharing
- Paste the BERT model in the main folder. Make sure that ./BERT_Review_finetuned/config.json exists
- Run either create_tips_absa.py for the new ABSA method or create_tips_baseline.py for the Baseline method
```sh
python create_tips_absa.py
```

```sh
python create_tips_baseline.py
```

- Optional: Change the hotel_of_interest variable to the desired hotel for which you want to create tips
- Optional: If you change the hotel of interest, don't forget to change to the corresponding right city. A dictionary of the city can be found in the two create_tips_absa.py or create.tips_baseline.py files


## Explanations
- absa_layer.py, bert.py, get-pip.py, glue_utils.py, seq_utils.py and bert_utils.py are from Li. et al. (2019)
- amenity_function.py finds the amenities of the hotels in the city-price stratum and hot one encodes them
- compare_aspects_function.py compares the information tokens of the different hotels to check if they appear significantly more
- cover_tokens_function.py selects the sentences to cover the information tokens
- create_synsets_function.py creates the synsets for the information tokens and compares them using Wu & Palmer similarity (Wu & Palmer, 1994).
- create_tips_absa.py is the main file to create tips using the proposed ABSA method
- create_tips_baseline.py is the main file to create tips using the Baseline method
- extract_aspects_function_absa.py extracts performs ABSA to extract information tokens (aspects) and sentiment using BERT. This is code from Li et al. (2019) modified to return aspects and sentiments and modified to work on the TipSelector Data.
- extract_tokens_baseline.py extracts information tokens using the baseline method
- LDA_function.py performs LDA to find similar hotels in combination with the amenities
- BERT_Review contains the post-trained BERT model from Xu et al. (2019) with the ABSA layer from Li et al. (2019)
- Output_sentences_absa and Output_sentences_baseline contain the end sentences for each hotel, for the ABSA method and the Baseline method respectiveily

## References

Li, X., Bing, L., Zhang, W., & Lam, W. (2019). Exploiting BERT for end-to-end aspect-based sentiment analysis. In W. Xu, A. Ritter, T. Baldwin, & A. Rahimi (Eds.), Proceedings of the 5th Workshop on Noisy User-generated Text (pp. 34–41). Hong Kong, China: Association for Computational Linguistics. Retrieved from https://doi.org/10.18653/ v1/D19-5505

Wu, Z., & Palmer, M. (1994). Verb semantics and lexical selection. arXiv preprint cmp- lg/9406033 .

Xu, H., Liu, B., Shu, L., & Yu, P. S. (2019). BERT post-training for review reading com- prehension and aspect-based sentiment analysis. In J. Burstein, C. Doran, & T. Solorio (Eds.), Proceedings of the 2019 Conference of the North American Chapter of the Associa- tion for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 2324–2335). Minneapolis, MN, USA,: Association for Computational Linguistics. Retrieved from https://doi.org/10.18653/v1/n19-1242

Zhu, D., Lappas, T., & Zhang, J. (2018). Unsupervised tip-mining from customer reviews. *Decision Support Systems, 107*, 116-124
