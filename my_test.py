from tkinter import *

def test_buttons():
    test_window = Tk()
    var = IntVar()
    test_window.title('1. Ваши сильные качества:')
    rbutton1 = Radiobutton(test_window, text= 'А) Рациональность.', variable=var, value=1) #Это начальные кнопки
    rbutton2 = Radiobutton(test_window, text= 'Б) Уравновешенность.', variable=var, value=2)
    rbutton3 = Radiobutton(test_window, text= 'В) Активность.', variable=var, value=3)
    rbutton4 = Radiobutton(test_window, text= 'Г) Позитивность.', variable=var, value=4)
    rbutton1.grid(column=0, row=0) 
    rbutton2.grid(column=1, row=0) 
    rbutton3.grid(column=2, row=0) 
    rbutton4.grid(column=3, row=0)
    but_one = Button(test_window, bg = 'red', text = 'PRESS') 
    but_three = Button(test_window, bg = 'blue', text = 'PRESS')
    but_four = Button(test_window, bg = 'yellow', text = 'PRESS')
    A = 0
    B = 0
    C = 0
    D = 0
    
    def counter(vary):  #Теперь это не событие, а функция
        x = vary.get()
        if x == 1:
            nonlocal A
            A = A + 1
        elif x == 2:
            nonlocal B
            B = B + 1
        elif x == 3:
            nonlocal C
            C = C + 1
        else:
            nonlocal D
            D = D + 1
    but_one.grid(column=4, row=2)

    def multy_config(w_list, t_list): 
        for x in range(len(w_list)):
            w_list[x].configure(text = t_list[x])
    q_2 = ['А) Нерешительность.', 'Б) Пассивность.', 'В) Непостоянство.', 'Г) Несерьезность.']
    q_3 = ['А) В пессимистичном.', 'Б) Спокойном.', 'В) Беспокойном, переменчивом.', 'Г) В хорошем.']
    q_4 = ['А) Необщителен.', 'Б) Малообщителен.', 'В) Средне общителен.', 'Г) Очень общителен.']
    def question_2(event):
        test_window.title('2. Ваши недостатки:')
        multy_config([rbutton1, rbutton2, rbutton3, rbutton4], q_2)
        but_one.destroy()
        but_three.grid(column=4, row=2)
        counter(var)
    but_one.bind('<Button-1>', question_2)

    def question_3(event):
        test_window.title('3. В каком настроении вы находитесь чаще всего?')
        multy_config([rbutton1, rbutton2, rbutton3, rbutton4], q_3)
        but_three.destroy()
        but_four.grid(column=4, row=2)
        counter(var)
        
    but_three.bind('<Button-1>', question_3)
    
    def question_4(event):
        but_four = Button (test_window, text='Закончить тест', command=test_window.destroy)
        but_four.grid(column=4, row=2)
        test_window.title('4. Насколько вы общительны?')
        multy_config([rbutton1, rbutton2, rbutton3, rbutton4], q_4)
        counter(var)
    but_four.bind('<Button-1>', question_4)
    test_window.mainloop()    
    return A, B, C, D

# print(test_buttons())  если тестить так, понятно, где не считает

def test_1(answers):
    mel = answers[0]
    fleg = answers[1]
    hol = answers[2]
    sang = answers[3]
    if mel == fleg == hol == sang:
        return 'Хм... Вы весьма гармоничная личность.'
    if sang is max(mel, fleg, hol, sang):
        return 'Вы преимущественно сангвиник.'
    elif fleg is max(mel, fleg, hol, sang):
        return 'Вы преимущественно флегматик.'
    elif hol is max(mel, fleg, hol, sang):
        return 'Вы преимущественно холерик.'
    else:
        return 'Вы преимущественно меланхолик.'
    
# print(test_1(test_buttons()))

def printer(answer):
    root = Tk()
    root.title('Ваш результат')
    root.geometry("500x100")
    Label(text=answer, font='Arial 14').pack()
    root.mainloop()

printer(test_1(test_buttons()))









