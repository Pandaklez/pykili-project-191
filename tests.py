from tkinter import *

    
def knopka(a, b, c, d, text): # задает окошко с вопросом и выбором значения
    root = Tk()
    var = IntVar()
    root.title(text)
    rbutton1 = Radiobutton(root, text= a, variable=var, value=1)
    rbutton2 = Radiobutton(root, text= b, variable=var, value=2)
    rbutton3 = Radiobutton(root, text= c, variable=var, value=3)
    rbutton4 = Radiobutton(root, text= d, variable=var, value=4)
    rbutton1.grid(column=0, row=0) 
    rbutton2.grid(column=1, row=0) 
    rbutton3.grid(column=2, row=0) 
    rbutton4.grid(column=3, row=0) 
    clicker = Button(root, text='click')
    clicker.grid(column=4, row=0)
    def counter(event):  
        global A
        A = 0
        global B
        B = 0
        global C
        C = 0
        global D
        D = 0
        x = var.get()
        if x == 1:
            A += 1
        elif x == 2:
            B += 1
        elif x == 3:
            C += 1
        else:
            D += 1
        rbutton1.destroy()
        rbutton2.destroy()
        rbutton3.destroy()
        rbutton4.destroy()
        clicker.destroy()
    clicker.bind('<Button-1>', counter)
    root.mainloop()
    return A, B, C, D

def test_1():
    result = []
    result.append(knopka('А) Рациональность.', 'Б) Уравновешенность.', 'В) Активность.', 'Г) Позитивность.', '1. Ваши сильные качества:'))
    result.append(knopka('А) Нерешительность.', 'Б) Пассивность.', 'В) Непостоянство.', 'Г) Несерьезность.', '2. Ваши недостатки:'))
    result.append(knopka('А) В пессимистичном.', 'Б) Спокойном.', 'В) Беспокойном, переменчивом.', 'Г) В хорошем.', '3. В каком настроении вы находитесь чаще всего?'))
    result.append(knopka('А) Необщителен.', 'Б) Малообщителен.', 'В) Средне общителен.', 'Г) Очень общителен.', '4. Насколько вы общительны?'))
    mel = result[0][0] + result[1][0] + result[2][0] + result[3][0]
    fleg = result[0][1] + result[1][1] + result[2][1] + result[3][1]
    hol = result[0][2] + result[1][2] + result[2][2] + result[3][2]
    sang = result[0][3] + result[1][3] + result[2][3] + result[3][3]
    if mel == fleg == hol == sang:
        return 'Hm... You are really harmonious person.'
    if sang is max(mel, fleg, hol, sang):
        return 'You are sanguine person.'
    elif fleg is max(mel, fleg, hol, sang):
        return 'You are phlegmatic person.'
    elif hol is max(mel, fleg, hol, sang):
        return 'You are choleric person.'
    else:
        return 'You are melancholiac person.'
    

print(test_1())
