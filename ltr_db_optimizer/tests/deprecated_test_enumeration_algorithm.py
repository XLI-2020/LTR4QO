import pytest
from enumerationAlgorithm.EnumerationAlgorithm import EnumerationAlgorithm

enum = EnumerationAlgorithm()

def test_extend_column_list():
    columns = ["O_CUSTKEY", "N_NATIONKEY"]
    tables = ["nation", "customer", "supplier", "region"]
    
    result = enum.extend_column_list(columns, tables)
    
    assert all(
        c in result for c in columns
    ), f"Not all 'start'-columns can be found in the result array: {result}"
    assert all(
        r in ['O_CUSTKEY', 'N_NATIONKEY', 'C_CUSTKEY', 'C_NATIONKEY', 'S_NATIONKEY'] for r in result
    ), f"There is an unknown column in result: {result}"
    assert all(
        r in result for r in ['O_CUSTKEY', 'N_NATIONKEY', 'C_CUSTKEY', 'C_NATIONKEY', 'S_NATIONKEY']
    ), f"Not all columns can be found in result: {result}"
    
def test_get_aggregate_info():
    sql_dict = {'Tables': [('nation', ['N_REGIONKEY', 'N_NATIONKEY']), ('region', ['R_REGIONKEY']),
                           ('supplier', ['S_ACCTBAL', 'S_NATIONKEY']), ('customer', ['C_ACCTBAL', 'C_MKTSEGMENT', 'C_NATIONKEY'])],
                'Joins': [('region', 'R_REGIONKEY', 'nation', 'N_REGIONKEY'), ('supplier', 'S_NATIONKEY', 'nation', 'N_NATIONKEY'),
                          ('customer', 'C_NATIONKEY', 'nation', 'N_NATIONKEY')],
                'Aggregation': {'Group By': [('customer', 'C_MKTSEGMENT')],
                                'Outputs': [('COUNT', None, '*'), ('AVG', 'supplier', 'S_ACCTBAL'),
                                            ('SUM', 'supplier', 'S_ACCTBAL'), ('SUM', 'customer', 'C_ACCTBAL')],
                                'Type': 'Group'},
                'Sort': [],
                'Top': None,
                'Filter': [],
                'Select': [('Group By', 'COUNT(*)'), ('Group By', 'AVG(S_ACCTBAL)'),
                           ('Group By', 'SUM(S_ACCTBAL)'), ('Group By', 'SUM(C_ACCTBAL)'),
                           ('Group By', 'C_MKTSEGMENT'), ('Group By', 'N_REGIONKEY')]}
    
    result = enum.get_aggregate_info(sql_dict)
    
    assert result["has_group"]
    assert len(result["group_by"]) == 1, "Wrong length of the result dictionary"
    assert ("customer", "C_MKTSEGMENT") == result["group_by"][0], f"Wrong first element in result dictionary['group_by']: {result['group_by']}"
    assert len(result['needed_fields']) == 4, f"Wrong length of needed_fields: {result['needed_fields']}"
    assert ('COUNT', None, '*', 'aggregate0') in result['needed_fields'], f"('COUNT', None, '*', 'aggregate0') was not found in result['needed_fields']: {result['needed_fields']}"
    assert ('AVG', 'supplier', 'S_ACCTBAL', 'aggregate1') in result['needed_fields'], f"('AVG', 'supplier', 'S_ACCTBAL', 'aggregate1') was not found in result['needed_fields']: {result['needed_fields']}"
    assert ('SUM', 'supplier', 'S_ACCTBAL', 'aggregate2') in result['needed_fields'], f"('SUM', 'supplier', 'S_ACCTBAL', 'aggregate2') was not found in result['needed_fields']: {result['needed_fields']}"
    assert ('SUM', 'customer', 'C_ACCTBAL', 'aggregate3') in result['needed_fields'], f"('SUM', 'customer', 'C_ACCTBAL', 'aggregate3') was not found in result['needed_fields']: {result['needed_fields']}"
    
    sql_dict ={'Tables': [('nation', ['N_REGIONKEY', 'N_NATIONKEY']), ('region', ['R_REGIONKEY']), 
                          ('supplier', ['S_ACCTBAL', 'S_NATIONKEY']), ('customer', ['C_ACCTBAL', 'C_NATIONKEY'])],
               'Joins': [('region', 'R_REGIONKEY', 'nation', 'N_REGIONKEY'), ('supplier', 'S_NATIONKEY', 'nation', 'N_NATIONKEY'),
                         ('customer', 'C_NATIONKEY', 'nation', 'N_NATIONKEY')],
               'Aggregation': {'Group By': [],
                               'Outputs': [('COUNT', None, '*'),
                                           ('AVG', 'supplier', 'S_ACCTBAL'),
                                           ('SUM', 'supplier', 'S_ACCTBAL'),
                                           ('SUM', 'customer', 'C_ACCTBAL')],
                               'Type': 'All'},
               'Sort': [],
               'Top': None,
               'Filter': [],
               'Select': [('Group By', 'COUNT(*)'), ('Group By', 'AVG(S_ACCTBAL)'),
                          ('Group By', 'SUM(S_ACCTBAL)'), ('Group By', 'SUM(C_ACCTBAL)')]}

    result = enum.get_aggregate_info(sql_dict)
    
    assert not result["has_group"], f"{result}"
    assert len(result['needed_fields']) == 4, f"Wrong length of needed_fields: {result['needed_fields']}"
    assert ('COUNT', None, '*', 'aggregate0') in result['needed_fields'], f"('COUNT', None, '*', 'aggregate0') was not found in result['needed_fields']: {result['needed_fields']}"
    assert ('AVG', 'supplier', 'S_ACCTBAL', 'aggregate1') in result['needed_fields'], f"('AVG', 'supplier', 'S_ACCTBAL', 'aggregate1') was not found in result['needed_fields']: {result['needed_fields']}"
    assert ('SUM', 'supplier', 'S_ACCTBAL', 'aggregate2') in result['needed_fields'], f"('SUM', 'supplier', 'S_ACCTBAL', 'aggregate2') was not found in result['needed_fields']: {result['needed_fields']}"
    assert ('SUM', 'customer', 'C_ACCTBAL', 'aggregate3') in result['needed_fields'], f"('SUM', 'customer', 'C_ACCTBAL', 'aggregate3') was not found in result['needed_fields']: {result['needed_fields']}"
    
    
def test_correct_sort():
    ordered_cols = [('DESC', 'R_NAME', 'region'), ('ASC', 'C_NATIONKEY', 'customer')]
    result = enum.test_and_correct_sort(ordered_cols)
    
    assert len(result["ascending"]) == len(result["columns"]), f"Lists of result have different length: {result}"
    assert len(result["ascending"]) == 2, f"Lists of result have unexpected length: {result}"
    assert "true" in result["ascending"] and "false" in result["ascending"], f"Wrong values in result['ascending']: {result['ascending']}"
    assert "R_NAME" in result["columns"] and "C_NATIONKEY" in result["columns"], f"Wrong values in result['columns']: {result['columns']}"
    
def test_split_cols():
    # Test "sort" 
    all_cols = ['aggregate0', 'R_NAME']
    part = ('sort', [], 'R_NAME', {'columns': ['R_NAME']}, False, False)
    ordered_cols = [('DESC', 'R_NAME', 'region')]
    
    result = enum.split_cols(all_cols, part, ordered_cols)
    
    right = result[0]
    left = result[1]
    
    assert not left[0] and not left[1], f"Second element in result was expected to be ([], None): {result}"
    assert 'aggregate0' in right[0] and 'R_NAME' in right[0], f"First element in result was expected to be (['aggregate0', 'R_NAME'], None): {result}"
    assert not right[1], f"First element in result was expected to be (['aggregate0', 'R_NAME'], None): {result}" 
    
    
    # Test compute scalar
    all_cols = ['aggregate0', 'R_NAME']
    part = ('compute_scalar', [], None,
            {'operations': [('IF_ELSE', {'if': ('compare', {'operator': 'EQ', 'column': 'tempcount0', 'value': '0'}),
                                         'then': ('const', {'CONSTVALUE': 'NULL'}), 'else': ('identifier', {'column': 'tempsum0'}), 'name': 'aggregate0'})]}, False, False)
    ordered_cols = None
    enum.agg_matcher = {'SUM(N_NATIONKEY)': 'aggregate0', 'tempsum0': 'N_NATIONKEY', 'aggregate0': ['tempsum0', 'tempcount0']}
    result = enum.split_cols(all_cols, part, ordered_cols)
    right = result[0]
    left = result[1]
    
    assert not left[0] and not left[1], f"Second element in result was expected to be ([], None): {result}"
    assert all(
        r in right[0] for r in ['tempsum0', 'tempcount0', 'R_NAME']
    ), f"First element in result was expected to be (['tempsum0', 'tempcount0', 'R_NAME'], None): {result}"
    assert all(
        r in ['tempsum0', 'tempcount0', 'R_NAME'] for r in right[0]
    ), f"First element in result was expected to be (['tempsum0', 'tempcount0', 'R_NAME'], None): {result}"
    assert not right[1], f"First element in result was expected to be (['tempsum0', 'tempcount0', 'R_NAME'], None): {result}" 
    
def test_unique_cols_calculation():
    # case 1
    right_unique_cols = []
    left_unique_cols = [1,2,3]
    left_col = 0
    right_col = 0
    calculated = enum.calculate_unique_cols(right_col, left_col, right_unique_cols, left_unique_cols)
    assert calculated[0] == left_unique_cols, "Calculated unique cols should be equal to left_unique_cols"
    # case 2
    right_unique_cols = [1,2,3]
    left_unique_cols = []
    calculated = enum.calculate_unique_cols(right_col, left_col, right_unique_cols, left_unique_cols)
    assert calculated[0] == right_unique_cols, "Calculated unique cols should be equal to right_unique_cols"    
    # case 3
    right_unique_cols = [0]
    left_unique_cols = [1]
    calculated = enum.calculate_unique_cols(right_col, left_col, right_unique_cols, left_unique_cols)
    assert calculated[0] == left_unique_cols, "Calculated unique cols should be equal to left_unique_cols"    
    # case 4
    right_unique_cols = [1]
    left_unique_cols = [0]
    calculated = enum.calculate_unique_cols(right_col, left_col, right_unique_cols, left_unique_cols)
    assert calculated[0] == right_unique_cols, "Calculated unique cols should be equal to right_unique_cols"     
    # case 5
    right_unique_cols = [0,1]
    left_unique_cols = [0]
    calculated = enum.calculate_unique_cols(right_col, left_col, right_unique_cols, left_unique_cols)
    assert calculated[0] == right_unique_cols, "Calculated unique cols should be equal to right_unique_cols"    
    # case 6
    right_unique_cols = [0]
    left_unique_cols = [0,1]
    calculated = enum.calculate_unique_cols(right_col, left_col, right_unique_cols, left_unique_cols)
    assert calculated[0] == left_unique_cols, "Calculated unique cols should be equal to left_unique_cols"         
    # case 7
    right_unique_cols = [1]
    left_unique_cols = [1]
    calculated = enum.calculate_unique_cols(right_col, left_col, right_unique_cols, left_unique_cols)
    assert calculated[0] == [], "Calculated unique cols should be empty"      
    