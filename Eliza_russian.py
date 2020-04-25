import re, random, tkinter, os, collections

def read(file):
    with open (file, encoding = 'utf-8') as f:
        text = f.read()
    text = text.replace('\n',' ').replace('  ',' ').replace('—',' ')
    for symbol in [',','"', ';',]:
        text = text.replace(symbol, '')
    text = text.split(' ')
    for word in text:
        if word == '':
            text.remove(word)
    return text


def tri(text): #Делает триграммы
    trigrammas = {}
    for i in range (len(text)-2):
        tri = text[i]+' '+text[i+1]+' '+text[i+2]
        if text[i] in trigrammas.keys():
            trigrammas[text[i]].append(tri)
        else:
            trigrammas[text[i]] = [tri]
    return trigrammas

def last_word(string): #Ищет последнее слово в строке
    last_word = re.search(r' (\S+$)', string)
    if last_word:
        return last_word.group(1)
    else:
        return ''

def two_last_words(string): #Ищет 2 последних слова в строке
    two_last_words = re.search(r' (\S+ \S+$)', string)
    if two_last_words:
        return two_last_words.group(1)
    else:
        return ''

def rand_replic(word, trigrammas):
    my_answ = word
    for x in range(40): #Делает текст из 40 слов
        if word[-1] not in '!?.': #Если текст - не квазипредложение
                if word in trigrammas.keys():
                    answ = collections.Counter(trigrammas[word]) #Выбирает самые частотные триграммы
                    if len(answ) >= 10:
                        i = random.randint(1, 9)
                    elif len(answ) >= 5:
                        i = random.randint(1, 4)
                    else:
                        i =1
                    answ = answ.most_common(i)
                    word = answ[i-1][0]
                    if answ[i-1] != 0: 
                        word = last_word(answ[i-1][0])
                        two_words = two_last_words(answ[i-1][0])
                        if two_words != '':
                            if two_words.split(' ')[0][-1] not in '!?.':
                                my_answ += ' '+two_words
                        else:
                            my_answ += ' '+two_words.split(' ')[0]
                            break
    my_answ += '.'
    my_answ = my_answ.replace('?.','?').replace('!.','!').replace(':.','.')
    my_answ = re.sub(r'\.+', '.', my_answ)
    return my_answ

text = read('alisa.txt') #Сюда надо вставить название файла с текстом
trigrammas = tri(text)
def random_phrase(ask, trigrammas):    
    for symbol in [',','"', ';']:
        ask = ask.replace(symbol, '')
    for symbol in ['!','?', '.']:
        ask = ask.replace(symbol, '')
    ask = ask.split(' ')
    possible_answ = ['!']
    for n in range(30):
        word = random.choice(ask)
        my_answ = rand_replic(word, trigrammas)
        if len(my_answ.split(' ')) <= 15 and len(my_answ.split(' '))>3: #выбирает более короткие тексты
            possible_answ.append(my_answ)
    if len(possible_answ) > 1:
        possible_answ.remove('!')
    answer = random.choice(possible_answ)
    let_one = answer[0]
    my_answer = let_one.capitalize()+answer[1:]
    return my_answer


def your_name(ask): # спрашивает имя, которое затем использунт в обращениях
    my_name=re.match("(((меня зовут)|(я))(\s))?(\S+)", ask.lower())
    if my_name:
        name=my_name.group(6)
        name = name.capitalize()
    else:
        name = ''
    return(name)

def your_replic(replic):
    your_replic=replic.lower()
    return your_replic

def start(name):
    if name:
        say_one = 'Приятно познакомиться, '+name+'! '
    else:
        say_one = 'Очень рада знакомству! '
    ask = random.choice(['Жизнь прекрасна?','У вас сейчас хорошее настроение?'])
    say = say_one+ask
    return say

def dict_answ(words, dict_of_words):
    dict_answer = ''
    for word in words.split(' '):
        if word in dict_of_words.keys():
            dict_answer = dict_of_words[word]
            break
            dict_answer = ''
    return dict_answer
            
def qu_answ(words, dict_of_words):
    qu_answ = dict_answ(words, dict_of_words)
    if qu_answ == '':
        qu_answ = random.choice(['Да!','Нет!'])
    return qu_answ
        
def reaction(words, list_of_words, list_of_words_two):
    for word in words.split(' '):
        if word in list_of_words:
            reaction_answer = random.choice(list_of_words_two)
            break
        else:
            reaction_answer = ''
    return reaction_answer

#def books(words):
    #if

def love(words):
    if ' не' in words:
        love_answer ='Почему нет?'
    else:
        if ('люблю') in words:
            topic = re.match(r'(.*?)(люблю)(.*)', words)
            love_answer = 'Я просто обожаю' + topic.group(3) + '!'#это вместо хобби, но вроде работает
        else:
            love_answer = random.choice(['Мне тоже :)','Как здорово!'])
    return love_answer 


def answer(replic):
    recepie = {'яблоки':'шарлотку','груши':'варенье','грибы':'соленье'} #это еда и блюда, можно дописать
    qu = {'почему':'По кочану.', 'когда':'После дождичка в четверг.', 'зачем':'Затем.',
          'сколько':'Столько.', 'где':'В Караганде.', 'кто':'Конь в пальто.'} #это ответы на вопросы с вопрос. словами
    sad = ('дождь', 'слякоть', 'ливень', 'сыро', 'жара') #это неприятности
    bad = ('Пакость!', 'Мерзость!', 'Какая жалость!', 'Фи!', 'Ох-ох', 'Ох!', 'Гадость!')#это "бранные слова"
    happy = ('солнечно', 'тепло', 'ясно')
    good = ('Прелесть!', 'Чудо!', 'Ах!', 'Это так мило!', 'Ах-ах!', 'Чудесно!', 'Прелестно!')#это приятные слова
    replic=replic.lower().replace('!', '')
    if replic == '':
        say = random.choice(["Не расслышала, повторите, пожалуйста!", "Не могли бы вы ответить понятнее?"])
    elif 'потому что' in replic:
        say = 'Спасибо, что поделились этим со мной! Доверие - это важно.'
    elif ('нравится' in replic or 'люблю' in replic):
        if not 'не' in replic:
            say = 'Можно приготовить '+dict_answ(replic, recepie)+'!'
            say = ''
        if say == '':
            say = love(replic)
    elif (replic.startswith('я')) and ('ненавижу' in replic):
        say = 'Поверьте, всё будет хорошо!'
    elif 'пока' in replic or 'до свидания' in replic or 'до встречи' in replic:
        say = random.choice(['Пока!','До свидания!','До встречи!','Возвращайтесь, я буду скучать!'])
    elif '?' in replic:
        say = qu_answ(replic, qu)
    else:
        say = reaction(replic, sad, bad)
        if say == '':
            say = reaction(replic, happy, good)
            if say == '':
                say = random_phrase(replic, trigrammas)
                if say == '!':
                    say = 'Вы любите котиков и пёсиков?'
    return(say)

def random_picture(photos_papka):
    files_list = os.listdir(photos_papka)
    choice = random.choice(files_list)
    photo = photos_papka + '\\' + choice
    return photo   


def main():
    root = tkinter.Tk()
    root.title('Eliza Doolittle')
    root.geometry('810x325')
    root["bg"] = 'pink'
    root.iconbitmap(r'icon.ico')
    eliza_say = tkinter.Label(root, text = 'Здравствуйте! Меня зовут Элиза, а вас?',
                              wraplength='300', width = '50', bg = 'pink')
    eliza_say.grid(column = 0, row = 1)
    you_say = tkinter.Entry(root, width = 50)
    you_say.grid(column = 1, row = 0)
    you_say_two = tkinter.Entry(root, width = 50)
    im = tkinter.PhotoImage(file='eliza_photo.ppm')
    my_photo = tkinter.Label(root, image = im, width = '400', height = '300', bg = 'pink')
    my_photo.grid(column = 1, row = 1)
    def hello(event):
        name = your_name(you_say.get())
        eliza_text = start(name)
        eliza_say.configure(text=eliza_text)
        you_say.destroy()
        you_say_two.grid(column = 1, row = 0)
    you_say.bind('<Return>', hello)
    
    def picture():
        print('!@!@')
        replic = you_say_two.get()
        you_say_two.delete('0', 'end')
        if 'кот' in replic:
            print('cat')
            #photo = random_picture(r'C:\НИУ ВШЭ\Программирование\Project\cat')
            #print(photo)
            #my_pict(my_photo, photo)
        #elif 'dog' in replic:
            #photo = random_picture(r'C:\Users\mi\Desktop\конспекты\python\ПРОЕКТ\photos\dog')
            #print(photo, root)
        #eliza_say.configure(text=random.choice(['Я уже говорила, что вы прекрасно выглядите? Я всё вижу ;)','Вы удивительно приятный собеседник!',
                                                #'С каждой минутой я всё сильнее рада знакомству с вами!', 'Давайте общаться чаще!']))

    def say(event):
        replic = you_say_two.get()
        you_say_two.delete('0', 'end')
        eliza_text = answer(replic.lower())
        eliza_say.configure(text=eliza_text)
        if eliza_text in ['Пока!','До свидания!','До встречи!','Возвращайтесь, я буду скучать!']:
            you_say_two.destroy()
    you_say_two.bind('<Return>', say)
    root.mainloop()
    
if __name__ == "__main__":
    main()
