import sympy as sp

def find_singularities():
    # 输入函数
    func_input = input("请输入一个数学函数 (例如：1/x, sin(x)/x): ")
    
    # 创建符号变量
    x = sp.symbols('x')
    
    # 解析输入的函数
    func = sp.sympify(func_input)
    
    # 查找奇异点
    singularities = sp.singularities(func, x)
    
    return singularities

def main():
    singularities = find_singularities()
    
    if singularities:
        print("该函数的奇异点是：")
        for point in singularities:
            print(point)
    else:
        print("该函数没有奇异点。")

if __name__ == "__main__":
    main()
