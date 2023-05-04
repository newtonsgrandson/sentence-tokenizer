from lang_detection import language_identification
from tokenizer import tokenizer

def main():


    while True:

        text = str(input("Text: "))        
        id, probabilty = language_identification(text).predict_lang()
        print(f"Detected language is {id}")
        tkr_en = tokenizer(text)
        tokenized_en = tkr_en.sentence_tokenizer()
        print(tokenized_en)
        print("x")

        if text == "exit":
            break




if __name__ == "__main__":
    main()