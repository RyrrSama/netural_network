
from numpy import average
from sklearn.covariance import log_likelihood
import torch


# Read data
def read_data(file_path, as_list=False):
    with open(file_path, encoding="utf-8") as f:
        return f.read().splitlines() if as_list else f.read()


# Get all unique words in the data
def get_unique_chars(data):
    return sorted(list(set(data)))


def str_to_int(char_list):
    """This function convert str to int and return a mapper dict"""
    return {char: i + 1 for i, char in enumerate(char_list)}


def int_to_str(str_mapper_dict):
    """This function will convert the int list to char list and return a mapper dict"""
    return {char_int: str_value for str_value, char_int in str_mapper_dict.items()}


def encode(char_list, mapper_dict):
    return [mapper_dict[char] for char in char_list]


def decode(int_list, mapper_dict):
    return "".join([mapper_dict[index] for index in int_list])


def get_pairchar_lookup_table(word_list, end_chr):
    list_unique_char = get_unique_chars("".join(word_list).lower())
    string_to_int = str_to_int(list_unique_char)
    string_to_int[end_chr] = 0
    muti_diarray = create_multi_di_array(len(string_to_int), len(string_to_int))

    for word in word_list:
        word = [end_chr] + list(word.lower()) + [end_chr]
        for char_01, char_02 in zip(word, word[1:]):
            muti_diarray[string_to_int[char_01], string_to_int[char_02]] += 1
    return muti_diarray


def get_pair_of_char_from_word(word_list, end_chr):
    pair_dict = {}
    for word in word_list:
        word = end_chr + list(word) + end_chr
        for char_01, char_02 in zip(word, word[1:]):
            pair_dict[char_01 + char_02] = pair_dict.get(char_01 + char_02, 0) + 1
    return pair_dict


def sort_dict_by_count(pair_dict, ascending=True):
    return sorted(pair_dict.items(), key=lambda kv: kv[1], reverse=not ascending)


def create_multi_di_array(number_of_rows, number_of_column, dtype=torch.int32):
    return torch.zeros((number_of_rows, number_of_column), dtype=dtype)

def get_bigram_model(data, propabilty_table,end_char="."):

    unique_char_list = get_unique_chars("".join(data).lower())
    stoi = str_to_int(unique_char_list)
    itos = int_to_str(stoi)
    stoi["."] = 0
    log_likelihood = 0.0
    number_pair = 0
    for name in data[:3]:
        name = name.lower()
        char = list(end_char) + list(name) + list(".")
        for char_01, char_02 in zip(char, char[1:]):
            index_01 = stoi[char_01]
            index_02 = stoi[char_02]
            prob = propabilty_table[index_01, index_02]
            logprob = torch.log(prob)
            log_likelihood += logprob
            number_pair+=1
            print(f"{char_01}{char_02}: {prob} {log_likelihood}")
    print(f"{logprob=}")
    nll = -log_likelihood
    average_nll = nll/number_pair
    print(f"{nll}")
    print(average_nll)
    return average_nll
            