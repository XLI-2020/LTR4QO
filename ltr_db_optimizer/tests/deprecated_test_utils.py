import ltr_db_optimizer.enumeration_algorithm.utils as utils

def test_unique():
    test_array_1 = ["customer", "supplier", "customer", "lineitem"]
    unique_array_1 = utils.unique(test_array_1)
    
    assert len(test_array_1) != len(unique_array_1), "Length of both arrays are the same but they are expected to be different"
    assert all(
        t in unique_array_1 for t in test_array_1        
    ), "There is an element in test_array_1 but not in unique_array_2"
    assert all(
        u in test_array_1 for u in unique_array_1
    ), "There is an unknown element in unique_array which cannot be found in test_array_1"
    
        
def test_overlap_function():
    # has no overlap
    l_1 = [1,2,3,4,5]
    l_2 = [6,7,8,9,0]
    assert not utils.test_overlap(l_1,l_2), f"There should be no overlap found between list 1 ({l_1}) and list 2 ({l_2})"
    l_2 = [1,7,8,9,0]
    assert utils.test_overlap(l_1,l_2), f"There should be an overlap found between list 1 ({l_1}) and list 2 ({l_2})"