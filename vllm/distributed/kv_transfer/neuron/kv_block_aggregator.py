# SPDX-License-Identifier: Apache-2.0
import logging

logger = logging.getLogger(__name__)


def aggregate_kv_blocks(lst1, lst2):
    """
    Chunk and align two lists of block ids.
    The lists must be the same length and contain no duplicates.
    
    
    To form a chunk:
    The two lists lst1 and lst2 must both:
        1. Have the same direction between adjacent elements (increasing 
        or decreasing).
        2. The values in each list must be consecutive integers, i.e.
        abs(a[i] - a[i-1]) == 1 and abs(b[i] - b[i-1]) == 1.
    
    for each pair of chunks, we sort the block ids in ascending order 
    to ease transfer.
    finally, merge adjacent chunks whenever possible to minimize the 
    number of chunks.
    
    The algorithm is guarantted to deliver unique results.
    The algorithm is guarantted to be commutative, i.e.
        aggregate_kv_blocks(a, b) == aggregate_kv_blocks(b, a)
    
    Complexity: O(n) where n is the length of the lists.

    Args:
        lst1: List of block ids from prefill node
        lst2: List of block ids from decode node

    Returns:
        Tuple of two lists of lists of block ids, each list of 
        lists is a chunk of block ids

    Example 1:
    a = [3,4,5,1,6,7,8,10,9]
    b = [3,4,5,6,7,8,9,11,10]
    out1, out2 = aggregate_kv_blocks(a, b)
    print(out1)  # [[3, 4, 5], [1], [6, 7, 8, 9, 10]]
    print(out2)  # [[3, 4, 5], [6], [7, 8, 9, 10, 11]]

    Example 2:
    a = [3,4,5,1,6,7,8,10,9]
    b = [3,2,5,6,7,8,9,11,10]
    out1, out2 = aggregate_kv_blocks(a, b)
    print(out1)  # [[3], [4], [5], [1], [6, 7, 8, 9, 10]]
    print(out2)  # [[3], [2], [5], [6], [7, 8, 9, 10, 11]]
    """
    if len(lst1) != len(lst2):
        raise ValueError("Block id lists must be the same length")

    # Check for duplicates
    if len(set(lst1)) != len(lst1):
        raise ValueError("Input lists must not contain duplicates.")
    if len(set(lst2)) != len(lst2):
        raise ValueError("Input lists must not contain duplicates.")

    result1, result2 = [], []
    temp1, temp2 = [lst1[0]], [lst2[0]]

    def get_dir(a, b):
        return 1 if b > a else -1

    for i in range(1, len(lst1)):
        a_prev, a_curr = lst1[i - 1], lst1[i]
        b_prev, b_curr = lst2[i - 1], lst2[i]

        dir_a = get_dir(a_prev, a_curr)
        dir_b = get_dir(b_prev, b_curr)

        is_consecutive = (abs(a_curr - a_prev) == 1
                          and abs(b_curr - b_prev) == 1 and dir_a == dir_b)

        if is_consecutive:
            temp1.append(a_curr)
            temp2.append(b_curr)
        else:
            if len(temp1) > 1 and get_dir(temp1[0], temp1[1]) == -1:
                temp1.reverse()
                temp2.reverse()
            result1.append(temp1)
            result2.append(temp2)
            temp1 = [a_curr]
            temp2 = [b_curr]

    if len(temp1) > 1 and get_dir(temp1[0], temp1[1]) == -1:
        temp1.reverse()
        temp2.reverse()
    result1.append(temp1)
    result2.append(temp2)

    # Merge adjacent chunks if possible
    def can_merge_pair(c1a, c2a, c1b, c2b):
        if not c1a or not c2a or not c1b or not c2b:
            return False

        da = c2a[0] - c1a[-1]
        db = c2b[0] - c1b[-1]

        return (da == 1 and db == 1)

    merged1, merged2 = [result1[0]], [result2[0]]
    for i in range(1, len(result1)):
        last1, last2 = merged1[-1], merged2[-1]
        curr1, curr2 = result1[i], result2[i]

        if can_merge_pair(last1, curr1, last2, curr2):
            merged1[-1] = last1 + curr1
            merged2[-1] = last2 + curr2
        else:
            merged1.append(curr1)
            merged2.append(curr2)

    return merged1, merged2
