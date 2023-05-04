from lang_detection import *
from tokenizer import tokenizer

def main():
    text_en = "like sharks snakes and spiders piranhas are widely feared although most people consider piranhas to be quite dangerous they are for the most part entirely harmless piranhas rarely feed on large animals they eat smaller fish and aquatic plants when confronted with humans piranhas first instinct is to flee not attack their fear of humans makes sense far more piranhas are eaten by people than people are eaten by piranhas if the fish are wellfed they wont bite humans"
    punctuated_text_en = "Like sharks, snakes, and spiders, piranhas are widely feared. Although most people consider piranhas to be quite dangerous, they are, for the most part, entirely harmless. Piranhas rarely feed on large animals; they eat smaller fish and aquatic plants. When confronted with humans, piranhas’ first instinct is to flee, not attack. Their fear of humans makes sense. Far more piranhas are eaten by people than people are eaten by piranhas. ,If the fish are well-fed, they won’t bite humans."

    id, probabilty = language_identification(text_en).predict_lang()
    print(f"Detected language is {id}")
    tkr_en = tokenizer(text_en)
    tokenized_en = tkr_en.sentence_tokenizer()
    test_en = tkr_en.text_to_test_data(punctuated_text_en)
    tkr_en.test_tokenized(tokenized_en, test_en)

    print("""-------------------------""")
    
    text_fr = "je mappelle jessica je suis une fille je suis française et jai treize ans je vais à lécole à nice mais jhabite à cagnessurmer jai deux frères le premier sappelle thomas il a quatorze ans le second sappelle yann et il a neuf ans mon papa est italien et il est fleuriste ma mère est allemande et est avocate mes frères et moi parlons français italien et allemand à la maison nous avons une grande maison avec un chien un poisson et deux chats"    
    punctuated_text_fr = "Je m’appelle Jessica. Je suis une fille, je suis française et j’ai treize ans. Je vais à l’école à Nice, mais j’habite à Cagnes-Sur-Mer. J’ai deux frères. Le premier s’appelle Thomas, il a quatorze ans. Le second s’appelle Yann et il a neuf ans. Mon papa est italien et il est fleuriste. Ma mère est allemande et est avocate. Mes frères et moi parlons français, italien et allemand à la maison. Nous avons une grande maison avec un chien, un poisson et deux chats."
    
    id, probabilty = language_identification(text_fr).predict_lang()
    print(f"Detected language is {id}")
    tkr_fr = tokenizer(text_fr)
    tokenized_fr = tkr_fr.sentence_tokenizer()
    test_fr = tkr_fr.text_to_test_data(punctuated_text_fr)
    tkr_fr.test_tokenized(tokenized_fr, test_fr)

    print("""-------------------------""")

    text_es = "yo vivo en granada una ciudad pequeña que tiene monumentos muy importantes como la alhambra aquí la comida es deliciosa y son famosos el gazpacho el rebujito y el salmorejo mi nueva casa está en una calle ancha que tiene muchos árboles el piso de arriba de mi casa tiene tres dormitorios y un despacho para trabajar el piso de abajo tiene una cocina muy grande un comedor con una mesa y seis sillas un salón con dos sofás verdes una televisión y cortinas además tiene una pequeña terraza con piscina donde puedo tomar el sol en verano me gusta mucho mi casa porque puedo invitar a mis amigos a cenar o a ver el fútbol en mi televisión además cerca de mi casa hay muchas tiendas para hacer la compra como panadería carnicería y pescadería"    
    punctuated_text_es = "Yo vivo en Granada, una ciudad pequeña que tiene monumentos muy importantes como la Alhambra. Aquí la comida es deliciosa y son famosos el gazpacho, el rebujito y el salmorejo. Mi nueva casa está en una calle ancha que tiene muchos árboles. El piso de arriba de mi casa tiene tres dormitorios y un despacho para trabajar. El piso de abajo tiene una cocina muy grande, un comedor con una mesa y seis sillas, un salón con dos sofás verdes, una televisión y cortinas. Además, tiene una pequeña terraza con piscina donde puedo tomar el sol en verano. Me gusta mucho mi casa porque puedo invitar a mis amigos a cenar o a ver el fútbol en mi televisión. Además, cerca de mi casa hay muchas tiendas para hacer la compra, como panadería, carnicería y pescadería."    
    
    id, probabilty = language_identification(text_es).predict_lang()
    print(f"Detected language is {id}")
    tkr_es = tokenizer(text_es)
    tokenized_es = tkr_es.sentence_tokenizer()
    test_es = tkr_es.text_to_test_data(punctuated_text_es)
    tkr_es.test_tokenized(tokenized_es, test_es)

    print("""-------------------------""")

    text_de = "familie müller plant ihren urlaub sie geht in ein reisebüro und lässt sich von einem angestellten beraten als reiseziel wählt sie mallorca aus familie müller bucht einen flug auf die mittelmeerinsel sie bucht außerdem zwei zimmer in einem großen hotel direkt am strand familie müller badet gerne im meer"    
    punctuated_text_de = "Familie Müller plant ihren Urlaub. Sie geht in ein Reisebüro und lässt sich von einem Angestellten beraten. Als Reiseziel wählt sie Mallorca aus. Familie Müller bucht einen Flug auf die Mittelmeerinsel. Sie bucht außerdem zwei Zimmer in einem großen Hotel direkt am Strand. Familie Müller badet gerne im Meer."    

    id, probabilty = language_identification(text_de).predict_lang()
    print(f"Detected language is {id}")
    tkr_de = tokenizer(text_de)
    tokenized_de = tkr_de.sentence_tokenizer()
    test_de = tkr_de.text_to_test_data(punctuated_text_de)
    tkr_de.test_tokenized(tokenized_de, test_de)

    print("""-------------------------""")

    text_it = "ciao mi chiamo carlo e ho 18 anni oggi vorrei parlarvi della mia tipica giornata mi alzo sempre alle e faccio una buona colazione con tè e biscotti"
    punctuated_text_it = "Ciao, mi chiamo Carlo e ho 18 anni. Oggi vorrei parlarvi della mia tipica giornata. Mi alzo sempre alle e faccio una buona colazione con tè e biscotti."    
        
    id, probabilty = language_identification(text_it).predict_lang()
    print(f"Detected language is {id}")
    tkr_it = tokenizer(text_it)
    tokenized_it = tkr_it.sentence_tokenizer()
    test_it = tkr_it.text_to_test_data(punctuated_text_it)
    tkr_it.test_tokenized(tokenized_it, test_it)

    print("""-------------------------""")

    unsupported_text = "A minha família é constituída por mim, pelo meu pai, a minha mãe, a minha irmã, os meus avós maternos e o meu avô paterno."
    id, probabilty = language_identification(unsupported_text).predict_lang()
    
if __name__ == "__main__":
    main()