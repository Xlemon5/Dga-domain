def n_grams(domen):
    grams_list = []
    n = [3, 4, 5]
    domen = domen.split('.')[0]
    for count_n in n:
        for i in range(len(domen)):
            if len(domen[i: count_n + i]) == count_n:
                grams_list.append(domen[i: count_n + i])
    return grams_list