# Dasar Text - Preprocessing

# Case Folding Lowercase
Kalimat = "Berikut ini adalah 5 negara dengan pendidikan terbaik di dunia adalah Korea Selatan, Jepang, Singapura, Hong Kong, dan Finlandia."
lower_case = Kalimat.lower()
print(lower_case)

# case Folding : Removing Number
import re 
kalimat = "Berikut ini adalah 5 negara dengan pendidikan terbaik di dunia adalah Korea Selatan, Jepang, Singapura, Hong Kong, dan Finlandia."

hasil = re.sub(r"\d+", "", kalimat)
print(hasil)


# Case Folding : Removing Punctuation
import string

kalimat = "Ini &adalah [contoh] kalimat? {dengan} tanda. baca?!!"
hasil = kalimat.translate(str.maketrans("","",string.punctuation))
print(hasil)

# Case Folding : Removing Whitespace
kalimat = " \t    ini kalimat contoh \t   \n"
hasil = kalimat.strip()

print(hasil)

# Separating Sentence with Split() Methode
kalimat = "rumah idaman adalah rumah yang bersih"
pisah = kalimat.split()
print(pisah)

# Tokenizing : Word Tokenizing Using NLTK Module
import nltk
from nltk.tokenize import word_tokenize 

kalimat = "Andi kerap melakukan transaksi rutin secara daring atau online."

tokens = nltk.tokenize.word_tokenize(kalimat)
print(tokens)

# Tokenizing with Case Folding
import string;
import nltk;
from nltk.tokenize import word_tokenize 
 
kalimat = "Andi kerap melakukan transaksi rutin secara daring atau online."
kalimat = kalimat.translate(str.maketrans('','',string.punctuation)).lower()

tokens = nltk.tokenize.word_tokenize(kalimat)

print(tokens)

# Frequency Distribution
import nltk; 
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import string;

kalimat = "Andi kerap melakukan transaksi rutin secara daring atau online. Menurut Andi belanja online lebih praktis & murah."
kalimat = kalimat.translate(str.maketrans('','',string.punctuation)).lower()

tokens = nltk.tokenize.word_tokenize(kalimat)
kemunculan = nltk.FreqDist(tokens)

print(kemunculan.most_common())

# Frequency Distribution Visualization with Matplotib
import matplotlib.pyplot as plt

kemunculan.plot(30, cumulative=False)

plt.show()

# Sentences Tokenizing Using NLTK Module
import nltk;
from nltk.tokenize import sent_tokenize

kalimat = "Andi kerap melakukan transaksi rutin secara daring atau online. Menurut Andi belanja online lebih praktis & murah."

tokens = nltk.tokenize.sent_tokenize(kalimat)

print(tokens)

#Filtetring using NLTK
import nltk;

from nltk.tokenize import sent_tokenize, word_tokenize;
from nltk.corpus import stopwords;
import string;

kalimat = "Andi dan icha kerap melakukan transaksi rutin secara daring atau online. Menurut Andi belanja online lebih praktis & murah."
kalimat = kalimat.translate(str.maketrans('','',string.punctuation)).lower()

tokens = word_tokenize(kalimat)
listStopword = set(stopwords.words('Indonesian'))

removed = []
for t in tokens:
    if t not in listStopword:
        removed.append(t)
 
print(removed)

# Filtering using Sastrawi : Stopword List
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
StopWordRemoverFactory

factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()
print(stopwords)

# Filtering using Sastrawi
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize
import string;
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
kalimat = "Andi kerap melakukan transaksi rutin secara daring atau online. Menurut Andi belanja online lebih praktis & murah."
kalimat = kalimat.translate(str.maketrans('','',string.punctuation)).lower()
stop = stopword.remove(kalimat)
tokens = nltk.tokenize.word_tokenize(stop)
print(tokens)
# output
# ['andi', 'kerap', 'transaksi', 'rutin', 'daring', 'online', 'andi', 'belanja', 'online', 'praktis', 'murah']

print(tokens)



# Steamming : Porter Stemming Algorithm using NLTK
from nltk.stem import PorterStemmer    
ps = PorterStemmer() 

kata = ["program", "programs", "programer", "programing", "programers"]

for k in kata:
    print(k, " : ", ps.stem(k))

# Stemming bahasa Indonesia using Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
 
kalimat = "Andi kerap melakukan transaksi rutin secara daring atau online. Menurut Andi belanja online lebih praktis & menyenangkan."

hasil = stemmer.stem(kalimat)

print(hasil);
