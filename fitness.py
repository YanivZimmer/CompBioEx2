def calc_freq(symble,freq_dict):
    if symble in freq_dict:
        return freq_dict[symble]
    return 0

def fitness(text,word_dict,pair_dict,letter_dict,lamda_arr):
    sum_letter = 0
    for l in text:
        sum_letter += calc_freq(l,letter_dict)
    #TODO:iterate over words
    sum_words = 0
    #TODO:iterate over pairs
    sum_pairs = 0
    return lamda_arr[2]*sum_letter+lamda_arr[1]*sum_pairs+lamda_arr[0]*sum_words