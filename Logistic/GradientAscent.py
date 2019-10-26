"""
函数说明：梯度上升算法测试函数
求函数：f(x)=-x^2+4x的极大值
Parameters:
    无
Returns：
    无
Modify:
    2018-8-13
"""
def Gradient_Ascent_test():
    def f_prime(x_old):
        return -2*x_old + 4 # f(x)的导数
    x_old = -1 # 初始值，给一个小于x_new的值
    x_new = 0 #梯度上升算法初始值，即从（0,0）开始
    alpha = 0.01# 步长，也就是学习速度，控制更新的幅度
    precision = 0.000000001 #精度，也就是更新阈值
    while abs(x_new-x_old)>precision:
        x_old = x_new
        x_new = x_new+alpha*f_prime(x_old)
    print(x_new)

"""
函数说明：main函数
"""
if __name__=='__main__':
    Gradient_Ascent_test()