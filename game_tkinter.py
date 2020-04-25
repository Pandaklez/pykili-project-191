from tkinter import *

def vocabulary(file):
    with open(file, encoding='utf-8') as f:
        text = f.read()
    words = text.split('\n')
    cities = []
    for i in words:
        if len(i) > 1:
            cities.append(i)
    for i in range(len(cities)):
        cities[i] = cities[i].replace('\u200e', '')
    return cities

def check_answer(programm_word, answer, all_rep, cities):
    letter = programm_word[-1]
    if letter == 'ё' or letter == 'ь':
        letter = answer[-2]
    if not answer.startswith(letter.upper()):
        return 'Я так не играю!'
    elif answer in all_rep:
        return 'Это слово уже было, я выиграла :р'
    elif answer not in cities:
        return 'Такого города нет!'
    else:
        return False

def reply(cities, all_rep, answer):
    letter = answer[-1]
    if letter == 'ё' or letter == 'ь':
        letter = answer[-2]
    for i in cities:
        if i.startswith(letter.upper()) and i not in all_rep:
            all_rep.append(i)
            return i
        
def loose_game(answer_new, reply_new, all_repl, voc):
        all_repl.append(answer_new)
        all_repl.append(reply_new)
        reply_new = reply(voc, all_repl, str(answer_new))
        if reply_new == None:
            return 'Поздравляю с победой!'

def continue_game(an_n, el_s, rep_n, all_r, voc):
        all_r.append(an_n)
        all_r.append(rep_n)
        el_s.configure(text=rep_n)
        answer_new = Entry(width = 70)
        all_r.append(rep_n)
        check_answer(rep_n, answer_new.get(), all_r, voc)
   
def main():
    root = Tk()
    root.title('Игра в города')
    root.geometry('600x300')
    root["bg"] = 'pink'
    eliza_say = Label(root, text='Начитаем игру! Москва',
                              wraplength='500', width = '70', bg = 'pink')
    eliza_say.grid(column = 0, row = 1)
    answer_new = Entry(width = 70)
    answer_new.grid(column = 1, row = 0)
    all_replies = []
    all_replies.append(answer_new)
    voc = vocabulary('cities.txt')
    def game(event):
        reply_new = reply(voc, all_replies, answer_new.get())
        if check_answer(eliza_say['text'], answer_new.get(), all_replies, voc) is False:
            loose_game(answer_new, reply_new, all_replies, voc)
            continue_game(answer_new, eliza_say, reply_new, all_replies, voc)
        else:
            eliza_say.configure(text=check_answer(eliza_say['text'], answer_new.get(), all_replies, voc))
        answer_new.delete(0, END)
    answer_new.bind('<Return>', game)
    root.mainloop()

if __name__ == "__main__":
    main()
        

