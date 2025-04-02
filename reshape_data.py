def reshape_data(data,window_width):
    n=len(data)
    print("该数据集有",n,"行")
    data_reshape=[]
    for i in range(n-window_width):
        data_reshape.append(data[i:i+window_width])
    print("经过重组之后，数据集变为了",len(data_reshape),"项")
    print("第一项为：")
    for it in data_reshape[0]:
        print(it)
    # print(data_reshape[0])
    return data_reshape