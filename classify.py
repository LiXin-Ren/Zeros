#testClasses = [17, 20, 27, 33, 74, 112, 134, 136, 148, 155]
#testClasses = ['ZJL'+str(i) for i in range(201, 231)]
attriFile = "TestAttributes.txt"    #只存了测试集中的201-231

def get_per_attributes():
    """
    获取每个类别的属性，返回一个列表
    """
    f = open(attriFile, "r")
    lines = f.readlines()

    per_attributes = []
    for line in lines:
        l = line.strip().split('\t')
        attributes = l[1:]
        num = []
        for i in range(30):
            num.append(float(attributes[i]))
        per_attributes.append(num)
    return per_attributes

attribute = get_per_attributes()    #30个测试类别的属性，list类型

def classfy(attri_pre, attris):
    res = []
    for i in range(30):
        dis = sum([(attri_pre[j] - attris[i][j])**2 for j in range(30)])
        res.append(dis)
    print(res)
    return 'ZJL'+str(res.index(min(res))+211)

test0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.8, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #211
test10 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3, 0.7, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #220
print(classfy(test10, attribute))
