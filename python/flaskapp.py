import flask
import psycopg2
import gensim
import pymystem3
import json
import os

app = flask.Flask(__name__)

# Возвращает список текстов, загруженных из базы данных
def load_texts_from_database(host, dbname, user, password):
    # Initialize connection to postgres    
    cursor = psycopg2.connect(host=host, dbname=dbname, user=user,
                                  password=password).cursor()
    
    # Get texts from database
    cursor.execute('select verse from verses')
    data = cursor.fetchall()
    texts = []
    for text in data:
        texts.append(text[0])
    
    return texts

# Функция преобразует список стихов в список списков токенов вида лемма_ЧАСТЬРЕЧИ
# Классификация частей речи в формате Mystem
# Возможен учёт стоп-слов
def preprocess(texts, stop_words=None):
    mystem = pymystem3.Mystem()
    preprocessed_texts = []
    
    for text in texts:
        preprocessed_text = []
        analized = mystem.analyze(text)
        
        for result in analized:
            if 'analysis' not in result:
                continue
            if result['analysis'] == []:
                continue
            lemma = result['analysis'][0]['lex']
            if stop_words is not None:
                if lemma in stop_words:
                    continue
            pos = result['analysis'][0]['gr'].split(',')[0].split('=')[0]
            preprocessed_text.append(lemma.lower() + '_' + pos)
            
        preprocessed_texts.append(preprocessed_text)
        
    return preprocessed_texts

# Создаёт и возвращает обученную модель векторизованных токенов
def create_model(preprocessed_texts, size=150, window=15, min_count=1, epochs=15):
    model = gensim.models.Word2Vec(preprocessed_texts,
                                   size=size, window=window, min_count=min_count)
    model.train(preprocessed_texts, total_examples=model.corpus_count,
                epochs=epochs)
    return model

# Возвращает список текстов из texts, которые удовлетворяют запросу по слову word
# topn - количество "похожих" слов, также используемых при поиске
# log - показать список слов, использвуемых при поиске (слово + похожие)
def search(word, model, texts, preprocessed_texts, topn=10, log=False):
    mystem = pymystem3.Mystem()
    analyzed = mystem.analyze(word)
    try:
        if 'analysis' not in analyzed[0]:
            raise KeyError('Incorrect search queue')
        if analyzed[0]['analysis'] == []:
            raise KeyError('Incorrect word')
        lemma = analyzed[0]['analysis'][0]['lex']
        pos = analyzed[0]['analysis'][0]['gr'].split(',')[0].split('=')[0]
        word = lemma + '_' + pos
        if log:
            print('Ищем: ' + word + ' ...')

        searched_texts = set()
        for i in range(len(preprocessed_texts)):
            if word in preprocessed_texts[i]:
                searched_texts.add(texts[i])
        if topn > 0:
            if log:
                print('А также:')
            for result in model.wv.most_similar(word, topn=topn):
                similar_word = result[0]
                if log:
                    print(similar_word)
                for i in range(len(preprocessed_texts)):
                    if similar_word in preprocessed_texts[i]:
                        searched_texts.add(texts[i])
            if log:
                print()
        return list(searched_texts)
    except KeyError:
        return ['Поиск не дал результатов']
    
# Дообучает модель новыми списками токенов
def update_model(model, preprocessed_texts, epochs=15):
    model.build_vocab(preprocessed_texts, update=True)
    model.train(preprocessed_texts, total_examples=len(preprocessed_texts),
                epochs=epochs)
    
# При запуске сервера загружаем модель или, если её нет, создаём и обучаем
verses = load_texts_from_database('localhost', 'kulyindlo', 'test_user', 'qwerty')
preprocessed_verses = preprocess(verses)
if os.path.isfile('./model'):
    model = gensim.models.Word2Vec.load('model')
else:
    model = create_model(preprocessed_verses)
    model.save('model')

# Ищет тексты по запросу ?q= и выдаёт результат в формате json
@app.route('/search')
def handle_search():
    query = flask.request.args.get('q')
    if query == None:
        return ''
    result = search(query, model, verses, preprocessed_verses)
    return json.dumps(result, ensure_ascii=False)

# Добавляет новый текст в модель и дообучает её
@app.route('/update', methods=['POST'])
def handle_update():
    if flask.request.method != 'POST':
        return ''
    verse = flask.request.form['text']
    preprocessed_verse = preprocess([verse])[0]
    verses.append(verse)
    preprocessed_verses.append(preprocessed_verse)
    update_model(model, [preprocessed_verse])
    model.save('model')
    return 'Success!'