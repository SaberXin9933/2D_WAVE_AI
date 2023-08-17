def split_list(input_list, size):
    """拆分list为若干个指定大小的子list"""
    return [input_list[i:i+size] for i in range(0, len(input_list), size)]