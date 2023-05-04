import fasttext

class language_identification:
    """

    Class object for Language Identification

    """

    def __init__(self, text):

        """

        Initializing class objects

        """
        self.text = text
        pretrained_lang_model = "lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)
        self.our_langs = ["en", "fr", "es", "de", "it"]

    def predict_lang(self):
        """

        Language prediction according to text

        """
        predictions = self.model.predict(self.text, k=1) # top 1 matching languages
        id = predictions[0][0].replace("__label__", "") # taking id according to returned string
        probabilty = float(predictions[1]) # it's probabilty

        if id in self.our_langs:
            return id, probabilty
        else:
            #raise TypeError(f"{id} is not supported")
            print(f"{id} is not supported")
            exit()
    
if __name__ == '__main__':
    language = language_identification()
    id_lang, probabilty = language.predict_lang("merhaba benim adım muhammed azad kılıçarslan")
    print(f"Your language is {id_lang} and it's probabilty is {probabilty}")