
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd
# import pickle
# model = pickle.load(open('finalized_model.pkl', 'rb'))
# cv=  pickle.load(open('finalized_model_tfidf.pkl', 'rb'))
import torch
import transformers
from transformers import AutoTokenizer, MobileBertForSequenceClassification
model_name = r'harshith20/Emotion_predictor'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MobileBertForSequenceClassification.from_pretrained(model_name)
emo_la={0:'sadness',1:'joy',2:'love',3:'anger',4:'fear',5:'surprise'}
# Release unneeded memory
torch.cuda.empty_cache()

# Enable cudnn auto-tuner
torch.backends.cudnn.benchmark = True
print('line-31')

contractions=pd.read_csv('contractions.csv',index_col='Contraction')
contractions.index = contractions.index.str.lower()
contractions.Meaning = contractions.Meaning.str.lower()
contractions_dict = contractions.to_dict()['Meaning']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
k=stopwords.words('english')

k=[i for i in k if i!='not']

def predict_emo(input_text):
    input_ids = tokenizer.encode(input_text, add_special_tokens=True, truncation=True, max_length=128)
    print(f'input_id :{input_ids}')
    # Convert input_ids to a PyTorch tensor
    input_tensor = torch.tensor([input_ids]).to(device)
    print('line-47')
    # Make predictions on the input
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs[0]

    # Get the predicted label
    print(f'output :{logits}')
    predicted_label = torch.argmax(logits, dim=1).item()
    print(emo_la[predicted_label])
    del  input_ids, outputs
    return emo_la[predicted_label]



def final_preprocessed(func):
    def inner(text):
        #text=[str(i).split(".") for i in text]
        text=pd.Series(text)
        #print(text)
        text=text.apply(lambda x:func(x))
        text=text.apply(lambda x:' '.join(x))
        #print('ajjkkkkkkffkfkvgkgj')
        # text = cv.transform(text)
        # output=[emo_la[s] for s in model.predict(text)]
        output=[predict_emo(i) for i in text]
        return output
    return inner    


urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
userPattern       = '@[^\s]+'
hashtagPattern    = '#[^\s]+'
alphaPattern      = "[^a-z0-9<>]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"
# Defining regex for emojis
smileemoji        = r"[8:=;]['`\-]?[)d]+"
sademoji          = r"[8:=;]['`\-]?\(+"
neutralemoji      = r"[8:=;]['`\-]?[\/|l*]"
lolemoji          = r"[8:=;]['`\-]?p+"
lemmatizer = WordNetLemmatizer()
@final_preprocessed
def text_cleaning(tweet):
    import re
    tweet = tweet.lower()

    # Replace all URls with '<url>'
    tweet = re.sub(urlPattern,'<url>',tweet)
    # Replace @USERNAME to '<user>'.
    tweet = re.sub(userPattern,'<user>', tweet)
    
    # Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    # Replace all emojis.
    tweet = re.sub(r'<3', '<heart>', tweet)
    tweet = re.sub(smileemoji, '<smile>', tweet)
    tweet = re.sub(sademoji, '<sadface>', tweet)
    tweet = re.sub(neutralemoji, '<neutralface>', tweet)
    tweet = re.sub(lolemoji, '<lolface>', tweet)

    for contraction, replacement in contractions_dict.items():
        tweet = str(tweet).replace(contraction, replacement)
    # Remove non-alphanumeric and symbols
        tweet = re.sub(alphaPattern, ' ', tweet)

    # Adding space on either side of '/' to seperate words (After replacing URLS).
        tweet = re.sub(r'/', ' / ', tweet)
        tweet = nltk.word_tokenize(tweet)
        tweet = [lemmatizer.lemmatize(sentence)  for sentence in tweet if sentence not in k]
        #if sentence not in stopwords.words('english')
    return tweet  
