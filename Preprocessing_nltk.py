# Preprocessing

# Load Library
import re
import string
import time
from copy import deepcopy

# Load dataset
import pandas as pd

df = pd.read_csv('Dataset_Sentimen_Emosi.csv')
df.head()
# print(df.head())

df = df.drop(['Emosi'], axis=1)
# print(df)
# df.info()

# # Preprocessing Dataset
# from ekphrasis.classes.preprocessor import TextPreProcessor
# from ekphrasis.classes.tokenizer import SocialTokenizer
# from ekphrasis.dicts.emoticons import emoticons

# text_processor = TextPreProcessor(
#     # terms that will be normalized
#     normalize=['email', 'percent', 'money', 'phone', 'user',
#         'time', 'date', 'number'],
#     # terms that will be annotated
#     #annotate={"hashtag", "allcaps", "elongated", "repeated",'emphasis', 'censored'},
#     annotate={"hashtag"},
#     fix_html=True,  # fix HTML tokens
    
#     # corpus from which the word statistics are going to be used 
#     # for word segmentation 
#     segmenter="twitter", 
    
#     # corpus from which the word statistics are going to be used 
#     # for spell correction
#     corrector="twitter", 
    
#     unpack_hashtags=True,  # perform word segmentation on hashtags
#     unpack_contractions=True,  # Unpack contractions (can't -> can not)
#     spell_correct_elong=False,  # spell correction for elongated words
    
#     # select a tokenizer. You can use SocialTokenizer, or pass your own
#     # the tokenizer, should take as input a string and return a list of tokens
#     tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
#     # list of dictionaries, for replacing tokens extracted from the text,
#     # with other expressions. You can pass more than one dictionaries.
#     dicts=[emoticons]
# )
# # panggil ekphrasis

# def bersih_data(text):
#     return " ".join(text_processor.pre_process_doc(text))

# # fungsi dari AMS 01-03. silakan cek bagaimana saya merubah menjadi fungsi

# def non_ascii(text):
#     return text.encode('ascii', 'replace').decode('ascii')

# def remove_space_alzami(text):
#     return " ".join(text.split())

# def remove_emoji_alzami(text):
#     return ' '.join(re.sub("([x#][A-Za-z0-9]+)"," ", text).split())

# def remove_tab(text):
#     return text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")

# def remove_tab2(text):
#     return re.sub('\s+',' ',text)

# def remove_rt(text):
#     return text.replace('RT'," ")

# def remove_mention(text):
#     return ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())

# def remove_incomplete_url(text):
#     return text.replace("http://", " ").replace("https://", " ")

# def remove_single_char(text):
#     return re.sub(r"\b[a-zA-Z]\b", "", text)

# def remove_excessive_dot(text):
#     return text.replace('..'," ")

# def change_stripe(text):
#     return text.replace('-'," ")

# def lower(text):
#     return text.lower()

# def remove_single_char(text):
#     return re.sub(r"\b[a-zA-Z]\b", "", text)

# def remove_excessive_dot(text):
#     return text.replace('..'," ")

# def lower(text):
#     return text.lower()

# def remove_whitespace_LT(text):
#     return text.strip()

# def remove_whitespace_multiple(text):
#     return re.sub('\s+',' ',text)

# def remove_punctuation(text):
#     remove = string.punctuation
#     remove = remove.replace("_", "") # don't remove hyphens
#     pattern = r"[{}]".format(remove) # create the pattern
#     return re.sub(pattern, "", text) 

# # hapus untuk <>
# def remove_number_eks(text):
#     return text.replace('<number>'," ")

# def remove_angka(text):
#     return re.sub(r"\d+", "", text) 

# def remove_URL_eks(text):
#     return text.replace('URL'," ").replace('url'," ")

# def space_punctuation(text):
#     return re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)

# # Lakukan fungsi pembersihan
# i = 0
# final_string = []
# s = ""
# for text in df['Tweet'].values:
#     filteredSentence = []
#     EachReviewText = ""
#     proc = remove_rt(text)
#     proc = lower(proc)
#     proc = change_stripe(proc)
#     proc = remove_emoji_alzami(proc)
#     proc = remove_tab(proc)
#     proc = remove_tab2(proc)
#     proc = non_ascii(proc)
#     proc = remove_incomplete_url(proc)
#     proc = remove_excessive_dot(proc)
#     proc = remove_whitespace_LT(proc)
#     proc = remove_whitespace_multiple(proc)
#     proc = remove_single_char(proc)
#     proc = space_punctuation(proc)
#     proc = remove_punctuation(proc)
#     proc = remove_space_alzami(proc)
#     proc = bersih_data(proc)
#     proc = remove_number_eks(proc)
#     proc = remove_angka(proc) 
#     proc = remove_URL_eks(proc)
#     EachReviewText = proc
#     final_string.append(EachReviewText)

# df["step01"] = final_string
# df.head(10)

# # Hapus Data Kosong
# df.info()
# df_hapus = df[~df['step01'].str.contains(" ")]
# df_hapus.info()
# df_hapus.head(10)
# df_new = df[~df.isin(df_hapus)].dropna()
# df_new.info()
# df_new

# # Normalisasi kata slang menjadi baku
# # token
# import nltk
# from nltk.tokenize import word_tokenize 
# def word_tokenize_wrapper(text):
#   return word_tokenize(text)
# df_new['tokens'] = df['step01'].apply(word_tokenize_wrapper)
# df_new.head(10)

# # normalized_word = pd.read_csv('kamus_clean.csv')
# #normalized_word_dict = {}

# for index, row in normalized_word.iterrows():
#     if row[0] not in normalized_word_dict:
#         normalized_word_dict[row[0]] = row[1] 
#         def normalized_term(document):
#     return [normalized_word_dict[term] if term in normalized_word_dict else term for term in document]
# df_new['final_tokens'] = df_new['tokens'].apply(normalized_term)
# i=0
# final_string_tokens = []
# for text in df_new['final_tokens'].values:
#     EachReviewText = ""
#     EachReviewText = ' '.join(text)
#     final_string_tokens.append(EachReviewText)
#     df_new["step02"] = final_string_tokens
#     df_new.head(10)
#     df_new.to_csv('clean_dataset.csv',sep=";")


