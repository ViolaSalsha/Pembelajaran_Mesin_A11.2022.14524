# # Case Folding Lowercase
# Kalimat = "Berikut ini adalah 5 negara dengan pendidikan terbaik di dunia adalah Korea Selatan, Jepang, Singapura, Hong Kong, dan Finlandia."
# lower_case = kalimat.lower()
# print(lower_case)

# # case Folding : Removing Number
# import re 
# kalimat = "Berikut ini adalah 5 negara dengan pendidikan terbaik di dunia adalah Korea Selatan, Jepang, Singapura, Hong Kong, dan Finlandia."

# hasil = re.sub(r"\d+", "", kalimat)
# print(hasil)


# # Case Folding : Removing Punctuation
# import string

# kalimat = "Ini &adalah [contoh] kalimat? {dengan} tanda. baca?!!"
# hasil = kalimat.translate(str.maketrans("","",string.punctuation))
# print(hasil)

# # Case Folding : Removing Whitespace
# kalimat = " \t    ini kalimat contoh \t   \n"
# hasil = kalimat.strip()

# print(hasil)

# # Separating Sentence with Split() Methode
# kalimat = "rumah idaman adalah rumah yang bersih"
# pisah = kalimat.split()
# print(pisah)

# # Tokenizing : Word Tokenizing Using NLTK Module
import nltk
# from nltk.tokenize import word_tokenize 

# kalimat = "Andi kerap melakukan transaksi rutin secara daring atau online."

# tokens = nltk.tokenize.word_tokenize(kalimat)
# print(tokens)

# Tokenizing with Case Folding
from nltk.tokenize import word_tokenize 
 
kalimat = "Andi kerap melakukan transaksi rutin secara daring atau online."
kalimat = kalimat.translate(str.maketrans('','',string.punctuation)).lower()

tokens = nltk.tokenize.word_tokenize(kalimat)

print(tokens)