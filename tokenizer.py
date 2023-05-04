from tokenizers import Tokenizer, models
import re
import requests
import time
import pandas as pd
from lang_detection import language_identification

text = "like sharks snakes and spiders piranhas are widely feared although most people consider piranhas to be quite dangerous they are for the most part entirely harmless piranhas rarely feed on large animals they eat smaller fish and aquatic plants when confronted with humans piranhas first instinct is to flee not attack their fear of humans makes sense far more piranhas are eaten by people than people are eaten by piranhas if the fish are well fed they wont bite humans"
punctuated_text = "Like sharks, snakes, and spiders, piranhas are widely feared. Although most people consider piranhas to be quite dangerous, they are, for the most part, entirely harmless. Piranhas rarely feed on large animals; they eat smaller fish and aquatic plants. When confronted with humans, piranhas’ first instinct is to flee, not attack. Their fear of humans makes sense. Far more piranhas are eaten by people than people are eaten by piranhas. ,If the fish are well-fed, they won’t bite humans."

"""

Punctuated:

Like sharks, snakes, and spiders, piranhas are widely feared. 
Although most people consider piranhas to be quite dangerous, they are, for the most part, entirely harmless. 
Piranhas rarely feed on large animals; they eat smaller fish and aquatic plants. 
When confronted with humans, piranhas’ first instinct is to flee, not attack. Their fear of humans makes sense. 
Far more piranhas are eaten by people than people are eaten by piranhas. ,
If the fish are well-fed, they won’t bite humans.

Unpunctuated:
like sharks snakes and spiders piranhas are widely feared 
although most people consider piranhas to be quite dangerous they are for the most part entirely harmless. 
piranhas rarely feed on large animals they eat smaller fish and aquatic plants
when confronted with humans, piranhas first instinct is to flee not attack their fear of humans makes sense 
far more piranhas are eaten by people than people are eaten by piranhas 
if the fish are well-fed, they wont bite humans

"""
API_URL = "https://api-inference.huggingface.co/models/kredor/punctuate-all"
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

class tokenizer:
    """

    Class object for English Tokenizer.

    """
    def __init__(self, text):
        """

        Initializing class objects
        
        Attributes
        text: type str, the sentence for which paragraph will be tokenized.

        """
        self.text = text

    def query(self, payload):
        """
        
        Request from api
        
        """
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    def sentence_tokenizer(self):
        """

        Tokenizer function.

        """
        while True:
            output = self.query({
                "inputs": self.text,
            })
            
            if not type(output) == dict:
                break

            if 'error' in list(output.keys()):
                print("Model is working, please wait for 20 sec.")
                time.sleep(3)

        output = pd.DataFrame(output)
        tokenized = []
        sentence = ""
        for i in output.iterrows():
            if i[1]["entity_group"] == ".":
                sentence += i[1]["word"] 
                tokenized.append(sentence)
                sentence = ""
            else:
                sentence += i[1]["word"] 
                sentence += " "

        return tokenized
    
    def normalize_string(self, string = str) :
        """

        Normalizing String

        """
        string = string.lower()
        string = re.sub(r'[^\w\s]', '', string)
        if string[0] == " ":
            string = string[1:len(string)]
        return string

    def text_to_test_data(self, text):
        """
        
        Normal text to test text for the tokenized text.
        
        """
        sentences = []
        sentence = ""
        for i in text:
            sentence += i
            if i in "!.;:?":
                sentences.append(sentence)
                sentence = ""
        
        sentences = list(pd.Series(sentences).agg(self.normalize_string))
        return sentences
    
    def test_tokenized(self, text_tokenized, test_tokenized):
        """
        
        Test function for tokenized text
        
        """
        for i in range(len(test_tokenized)):
            if text_tokenized[i] == test_tokenized[i]:
                print("True " + str(i)+ "\t" + test_tokenized[i])
            else:
                print("False " + str(i)+ "\t" + "Tokenized: " + text_tokenized[i] + "\t" + "Test: " + test_tokenized[i])

            if (i + 2) > len(text_tokenized):
                break


def main():
    global text
    global punctuated_text

    print(f"Detected language is {language_identification(text).predict_lang()}")
    tkr = tokenizer(text)
    tokenized = tkr.sentence_tokenizer()
    test = tkr.text_to_test_data(punctuated_text)
    tkr.test_tokenized(tokenized, test)

def main_normalizer():
    text = "I just returned from the greatest summer vacation! It was so fantastic, I never wanted it to end. I spent eight days in Paris, France. My best friends, Henry and Steve, went with me. We had a beautiful hotel room in the Latin Quarter, and it wasn’t even expensive. We had a balcony with a wonderful view."
    tkr = tokenizer(text)
    print(tkr.normalize_string(text))


if __name__ == "__main__":
    main_normalizer()
