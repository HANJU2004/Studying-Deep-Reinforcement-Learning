import keyboard
import plotly
import tkinter #导入Tkinter模块
root = tkinter.Tk()#生成一个主窗口对象
#实例化标签组件
label= tkinter.Label(root, text="Python, tkinter!")
label.pack()#将标签添加到窗口中
button1 = tkinter.Button(root, text="按钮1")#创建按钮1
button1.pack(side=tkinter.LEFT)#将按钮1添加到窗口中
button2 = tkinter.Button(root, text="按钮2")#创建按钮2
button2.pack(side=tkinter.RIGHT)#将按钮2添加到窗口中
root.mainloop()#进入消息循环