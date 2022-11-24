#Importing libraries
import re
import nltk
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer  # used for lemmatizer
import string
import contractions
from spellchecker import SpellChecker
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import demoji
import pickle
import streamlit as st
import compress_fasttext
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


small_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load("cc.en.300.compressed.bin")
model=pickle.load(open('C:/Users/mansi/Desktop/Study/Hate Speech Detection/models/voting.pkl','rb'))


# ======================================================================================================================
# Creating list of Stop-words (pre-processing)
# ======================================================================================================================
new_stopwords=['say', 'get', 'go', 'know', 'may', 'need', 'make', 'see', 'want', 'come', 'take', 'use','life','money',
               'little','even','head','right','eat','laugh','well','red','bad','best','year','today','watch','win','play',
               'new','game','good','would', 'can', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'may',
           'also', 'across', 'among', 'beside', 'yet', 'within', 'mr', 'bbc', 'image', 'getty','woman','boy','guy'
           'de', 'en', 'caption', 'copyright', 'something', 'tag', 'wait', 'set', 'put', 'add', 'post', 'give', 'way', 'check', 'think',
          'www', 'must', 'look', 'call', 'minute', 'com', 'thing', 'much', 'happen','still','tell','talk','never','every,'
          'quarantine', 'day', 'time', 'week', 'amp', 'find','None','man','girl','really','real','people','love','like','let','back' ]
stop_words = set(list(stopwords.words('english')) + ['"', '|'] + new_stopwords)

# ======================================================================================================================
# Creating list of Emoticons (pre-processing)
# ======================================================================================================================
# Happy Emoticons
emoticons_happy = {':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ':-D', ':D', '8-D',
               '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P',
               ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3'}

# Sad Emoticons
emoticons_sad = {':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<', ':-[', ':-<', '=\\', '=/',
             '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c', ':c', ':{', '>:\\', ';('}

# Combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)
# ======================================================================================================================
# Removing links (pre-processing)
# ======================================================================================================================
def strip_links(text):
    all_links_regex = re.compile('http\S+|www.\S+', re.DOTALL)
    text = re.sub(all_links_regex, '', text)
    return text

# ======================================================================================================================
# Removing Punctuations (pre-processing)
# ======================================================================================================================
def remove_punctuation(text):
    text = re.sub(r'@\S+', '', text)  # Delete Usernames

    # remove punctuation from each word (Replace hashtags with space, keeping hashtag context)
    text = re.sub(r'#', '', text)  # Delete the hashtag sign

    for separator in string.punctuation:
        if separator not in ["'"]:
            text = text.replace(separator, ' ')

    return text

# ======================================================================================================================
# Removing Emojis (pre-processing)
# ======================================================================================================================

def remove_emoji(text):
    text= demoji.replace(text, "")
    return text

# ======================================================================================================================
# Removing Contractions (pre-processing)
# ======================================================================================================================

def get_contractions(text):
    commonSMS = {
    "Ain’t" : "Am not",
    "Wanna" : "Want to",
    "Whatcha" : "What have you",
    "Kinda" : "Kind of",
    "Sorta" : "Sort of",
    "Outta" : "Out of",
    "Alotta" : "A lot of",
    "Lotsa" : "Lots of",
    "Mucha" : "Much of",
    "Cuppa" : "Cup of",
    "Dunno" : "Don’t know",
    "Lemme" : "Let me",
    "Gimme" : "Give me",
    "Tell’em" : "Tell them",
    "Imma" : "I am going to",
    "Gonna" : "Going to",
    "Needa" : "Need to",
    "Oughta" : "Ought to",
    "Hafta" : "Have to",
    "Hasta" : "Has to",
    "Usta" : "Used to",
    "Supposta" : "Supposed to",
    "Gotta" : "Got to",
    "Cmon" : "Come on",
    "Ya" : "You",
    "Shoulda" : "Should have",
    "Shouldna" : "Should not have",
    "Wouldna" : "Would not have",
    "She’da" : "She would have",
    "Coulda" : "Could have",
    "Woulda" :"Would have",
    "Mighta" : "Might have",
    "Mightna" : "Might not have",
    "Musta" : "Must have",
    "Mussna" : "Must not have",
    "Dontcha" : "Do not you",
    "Wontcha" : "Would not you",
    "Whatcha" : "What are you",
    "Betcha" : "Bet you",
    "Gotcha" : "Got you",
    "D’you" : "Do you",
    "Didntcha" : "Did not you",
    "Dija" : "Did you",
    "S’more" : "Some more",
    "Layder" : "Later",
    "R": "are",
    "N":"and",
    "D":'the',
    "BRB":"Be right back",
    "IKR":"I know, right",
    "ILY":"I love you",
    "LMFAO":"Laughing my freaking ass off",
    "NVM": "Never mind",
    "OFC": "Of course",
    "ROFL":"Rolling on the floor laughing",
    "SMH": "Shaking my head",
    "STFU": "Shut the fuck up",
    "YOLO": "You only live once",
    "MMB":"Message me back",
    "YNT":"Why not",
    "BW":"Between",
    "TC":"Take care",
    "MU":"Miss you",
    "S2R":"Send to receive",
    "NVM":"Never mind",
    "CTN":"Can’t talk now",
    "B4":"Before",
    "FTW":"For the win",
    "HW":"Homework",
    "W8":"Wait",
    "PC":"Personal computer",
    "ITT":"In this thread",
    "RBTL":"Read between the lines",
    "ETA":"Estimated time of arrival",
    "XOXO":"Hugs and kisses",
    "AFK":"Away from keyboard",
    "BuBye":"Bye Bye",
    "DIY":"Do it yourself",
    "MW":"On my way",
    "SD":"Sweet dreams",
    "YW":"You are welcome",
    "RL":"Real life",
    "SRY":"Sorry",
    "DIKU":"Do I know you",
    "IDGI":"I do not get it",
    "IDC":"I do not care",
    "IDK":"I do not know",
    "CFY":"Calling for you",
    "AAMOF":"As a matter of fact",
    "TYT":"Take your time",
    "TY":"Thank you",
    "GG":"Good game",
    "IRL":"In real life",
    "GJ":"Good job",
    "POV":"Point of view",
    "R8":"Right",
    "BTW":"By the way",
    "SU":"Shut up",
    "NC":"No comment",
    "SEC":"Second",
    "IMO":"In my opinion",
    "JK":"Just kidding",
    "KK":"Okay cool",
    "PPL":"People",
    "GTG":"Got to go",
    "NP":"No problem",
    "ROFL":"Rolling on the floor laughing",
    "RIP":"Rest in peace",
    "SMH":"Shaking my head",
    "PLZ":"Please",
    "RT":"Real time",
    "CYL":"Call you later",
    "GM":"Good morning",
    "GR8":"Great",
    "YOLO":"You only live once",
    "GN":"Goodnight",
    "WD":"Well done",
    "TTYS":"Talk to you soon",
    "BD":"Big deal",
    "GL":"Good luck",
    "L8R":"Later",
    "TTYL":"Talk to you later",
    "TMI":"Too much information",
    "IM":"Instant message",
    "ASIC":"As soon as I can",
    "TCO":"Taken care of",
    "BBIAB":"Be back in a bit",
    "B4N" :"Bye for Now",
    "HU":"Hug you",
    "QT":"Cutie",
    "MSG":"Message",
    "LOL":"laugh out loud",
    "ZZZ":"Sleeping",
    "IC":"I see",
    "JJ":"Just joking",
    "F2F":"Face to face",
    "BRB":"Be Right Back",
    "CTN":"Can not talk now",
    "TTYN":"Talk to you never",
    "BFF":"Best Friends Forever",
    "GBTW":"Get back to work",
    "LMAO":"laughing my ass off",
    "BC":"Because",
    "PLS":"Please",
    "NOOB":"Newbie",
    "WTF":"What the fuck",
    "CU":"See you",
    "FAB":"Fabulous",
    "THX":"Thanks",
    "CUL":"See you later",
    "COZ":"Because",
    "CUZ":"Because",
    "CAUSE":"Because",
    "CYA":"See You",
    "Y":"Why",
    "TXT":"Text",
    "KU":"Kiss you",
    "FYI":"For your information",
    "OOO":"Out of office",
    "FAQ":"Frequently asked questions",
    "LU":"Love you",
    "AKA":"Also known as",
    "THO":"Though",
    "BAU":"Business as usual",
    "HBU":"How about you",
    "LMAO":"Laughing my ass off",
    "AFAIK" :"As far as I know",
    "BA3":"Battery",
    "GMV":"Got my vote",
    "RT":"Retweet",
    "IMHO":"In my humble opinion",
    "HTH":"Here to help",
    "BF":"Boyfriend",
    "PC":"Personal computer",
    "L8":"Late",
    "ASAP":"As soon as possible",
    "GONNA":"Going to",
    "GUNNA":"Going to",
    "OMG":"Oh my God",
    "LAM":"Leave a message",
    "NTN":"No thanks needed",
    "SS":"So sorry",
    "M8":"Mate",
    "2MORO":"Tomorrow",
    "LNG":"Long",
    "pic":"picture",
    "OMG":"Oh my god",
    "GAL":"Girl",
    "DND":"Do not disturb",
    "10Q":"Thank you",
    "2B":"To be",
    "4EVA":"Forever",
    "2MOR" :"Tomorrow",
    "YT":"YouTube",
    "utube": "Youtube",
    "der":"there",
    "wrk":"work",
    "tv":"television",
    "lol":"Laugh out loud",
    "4got":"Forgot",
    "yr":"year",
    "hr":"hour",
    "b4":"before",
    "bout":"about",
    "c":"see",
    "dat":"that",
    "tellin":"telling"
    }
    new_dict = dict((k.lower(), v.lower()) for k, v in  commonSMS.items())
    text_decontracted = []

    for word in text.split():
        if word in new_dict:
            word = new_dict[word]
        text_decontracted.append(word)

    text = ' '.join(text_decontracted)
    return text

# ======================================================================================================================
# Removing Spelling mistakes (pre-processing)
# ======================================================================================================================

# '''To perform spelling mistake correction, you first need to make sure the word
# is not absurd or from slang like, caaaar, amazzzing etc. with repeated alphabets.
# '''
def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

# '''use Pyspellchecker library from Python for correcting spellings'''

def spell_checker(text):
    spell = SpellChecker()
    text=' '.join(str(spell.correction(w)) for w in text.split())
    return text

# ======================================================================================================================
# Removing Non-alphanumeric words (pre-processing)
# ======================================================================================================================

# function to keep only alpharethmetic values
def only_alpha(tokenized_text):
    text_alpha = []
    for word in tokenized_text:
        word_alpha = re.sub('[^a-z A-Z]+', ' ', word)
        text_alpha.append(word_alpha)
    return text_alpha

# ======================================================================================================================
# Applying Lemmatization(pre-processing)
# ======================================================================================================================

# convert POS tag to wordnet tag in order to use in lemmatizer
lemmatizer = WordNetLemmatizer()

def nltk_pos_tagger(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
#lemmatizing
def lemmatizing(tokenized_text):

    nltk_tagged = nltk.pos_tag(tokenized_text)
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    lemma_list = []

    for word, tag in wordnet_tagged:
        if tag is None:
            lemma_list.append(word)
        else:
            lemma_list.append(lemmatizer.lemmatize(word, tag))
    return lemma_list

def preprocessing(text):
        # lower text
        text = text.lower().strip()

        # remove punctuations and links
        text = remove_punctuation(strip_links(text))

        # remove emails
        text = re.sub('\S*@\S*\s?', '', text)

        # remove rt and via in case of tweet data
        text = re.sub(r"\b(rt|RT)\b", "", text)
        text = re.sub(r"\b(via|VIA)\b", "", text)
        text = re.sub(r"\b(it|IT)\b", "", text)
        text = re.sub(r"\b(btu|BTu)\b", "", text)
        text = re.sub(r"\b(bt|BT )\b", "", text)

        # format contractions without apostrophe in order to use for contraction replacement
        text = re.sub(r"\b(s|'s)\b", " is ", text)
        text = re.sub(r"\b(ve|'ve)\b", " have ", text)
        text = re.sub(r"\b(nt|'nt| 't)\b", " not ", text)
        text = re.sub(r"\b(re|'re)\b", " are ", text)
        text = re.sub(r"\b(d|'d)\b", " would ", text)
        text = re.sub(r"\b(ll|'ll)\b", " will ", text)
        text = re.sub(r"\b(m|'m)\b", " am", text)

        # replace consecutive non-ASCII characters with a space
        # examples=भारत (used for websites in India), 网络 (the .NET equivalent in China),קום(the .COM equivalent in Hebrew)
        #           இந்தியா (meaning ‘Tamil’ for India, which is a language spoken in parts of India)
        #
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        # remove emojis from text
        # text = self.emoji_pattern.sub(r'', text)
        text = remove_emoji(text)

        # substitute contractions with full words
        # english contractions
        text = contractions.fix(text)
        # Most common internet contractions
        text = get_contractions(text)

        # Check for spelling corrections
        text = reduce_lengthening(text)
        text = spell_checker(text)

        tokenized_text = word_tokenize(text)

        # remove all non alpharethmetic values
        tokenized_text = only_alpha(tokenized_text)

        # lemmatize / stem words
        lemmatized_text = lemmatizing(tokenized_text)

        filtered_text = []
        # looping through conditions
        for word in lemmatized_text:
            word = word.strip()
            # check tokens against stop words, emoticons and punctuations
            # biggest english word: Pneumonoultramicroscopicsilicovolcanoconiosis (45 letters)
            if (word not in stop_words and word not in emoticons and word not in string.punctuation
                    and not word.isspace() and len(word) > 1 and len(word) < 46):
                filtered_text.append(word)

        return filtered_text

##Word 2 Vec
def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += small_model[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec

def predict(text):
    text = preprocessing(text)
    wordvec_arrays = word_vector(text, 300)
    y_pred = model.predict(wordvec_arrays)

    if y_pred == 0:
        return "Hateful Tweet!"
    if y_pred == 1:
        return "Offensive Language!"
    if y_pred == 2:
        return "Neither Offensive Nor Hate."
# ======================================================================================================================
# Streamlit App
# ======================================================================================================================
def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url("https://img.freepik.com/free-vector/hand-painted-watercolor-pastel-sky-background_23-2148902771.jpg?w=2000");
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
set_bg_hack_url()
def header(prediction):
    st.markdown(f'<p style="color:#FE0303 ;font-size:24px;font-weight:bold;text-align: center;">{prediction}</p>', unsafe_allow_html=True)

st.markdown("""
    <style>
    .stTextArea [data-baseweb=base-input] {
        background-color:#FADBD8 ;
        -webkit-text-fill-color: #4A235A ;
    }

    .stTextArea > label {font-size:20px; font-weight:bold; color:#FF5733; }
    
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color:#FF5733;'>Twitter Hate/Offensive Speech Detection</h2>", unsafe_allow_html=True)
st.write("<h5 style='text-align: center; color:#222225;'>This classifier is taught to distinguish between three categories of texts: Hateful Speech', 'Offensive Language', and 'Neither Offensive Nor Hateful'.</h5>", unsafe_allow_html=True)
st.write("<h5 style='text-align: center; color:#222225;'>The widespread usage of social media, blogging, and other social communication services has been more apparent in recent years. Users of these services are exposed to content made by other people. The majority of the time, that content may include derogatory or abusive language that could target or insult particular user groups without offering helpful information.</h5>", unsafe_allow_html=True)

input_tweet = st.text_area(label="Enter the tweet: ")


if st.button('Predict'):
    if input_tweet != "":
        prediction=predict(input_tweet)
        header(prediction)
    else:
         header("Please write a tweet to detect!")

hate_png="hate.png"
offensive_png="offensive.png"

col1, col2 = st.columns(2)

with col1:
   header("HATE WordCloud")
   st.image(hate_png)

with col2:
   header("OFFENSIVE WordCloud")
   st.image(offensive_png)

neither_png="neither.png"
header("Neither HATE Nor OFFENSIVE")

col1, col2, col3 = st.columns([1.5,5,2])
col2.image(neither_png, width=500)