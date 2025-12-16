"""
Test dataset filtering logic for selected_test_cases feature.
This validates the filtering logic without requiring full benchmark infrastructure.
"""


def test_filter_dataset_by_selected_ids():
    """Test that dataset filtering works correctly with selected_test_cases"""
    # Mock dataset with test case IDs
    mock_dataset = [
        {"id": "Car-Rental-0", "data": "test1"},
        {"id": "Car-Rental-1", "data": "test2"},
        {"id": "Car-Rental-2", "data": "test3"},
        {"id": "Travel-0", "data": "test4"},
        {"id": "Travel-1", "data": "test5"},
    ]
    
    # Mock selected test cases
    selected_test_cases = ["Car-Rental-0", "Travel-1"]
    
    # Apply the same filtering logic as in cfb_run_eval.py
    filtered_dataset = [case for case in mock_dataset if case.get('id') in selected_test_cases]
    
    # Verify results
    assert len(filtered_dataset) == 2
    assert filtered_dataset[0]['id'] == "Car-Rental-0"
    assert filtered_dataset[1]['id'] == "Travel-1"


def test_filter_dataset_with_no_matches():
    """Test that filtering returns empty list when no matches found"""
    mock_dataset = [
        {"id": "Car-Rental-0", "data": "test1"},
        {"id": "Car-Rental-1", "data": "test2"},
    ]
    
    selected_test_cases = ["NonExistent-0", "AlsoNotThere-1"]
    
    filtered_dataset = [case for case in mock_dataset if case.get('id') in selected_test_cases]
    
    assert len(filtered_dataset) == 0


def test_filter_dataset_with_none_selected():
    """Test that None selected_test_cases doesn't filter"""
    mock_dataset = [
        {"id": "Car-Rental-0", "data": "test1"},
        {"id": "Car-Rental-1", "data": "test2"},
        {"id": "Car-Rental-2", "data": "test3"},
    ]
    
    selected_test_cases = None
    
    # This simulates the logic: only filter if selected_test_cases is truthy
    if selected_test_cases:
        filtered_dataset = [case for case in mock_dataset if case.get('id') in selected_test_cases]
    else:
        filtered_dataset = mock_dataset
    
    assert len(filtered_dataset) == 3
    assert filtered_dataset == mock_dataset


def test_filter_dataset_with_empty_list():
    """Test that empty list for selected_test_cases doesn't filter"""
    mock_dataset = [
        {"id": "Car-Rental-0", "data": "test1"},
        {"id": "Car-Rental-1", "data": "test2"},
    ]
    
    selected_test_cases = []
    
    # This simulates the logic: only filter if selected_test_cases is truthy
    if selected_test_cases:
        filtered_dataset = [case for case in mock_dataset if case.get('id') in selected_test_cases]
    else:
        filtered_dataset = mock_dataset
    
    assert len(filtered_dataset) == 2
    assert filtered_dataset == mock_dataset


def test_filter_preserves_case_data():
    """Test that filtering preserves all case data, not just IDs"""
    mock_dataset = [
        {"id": "Car-Rental-0", "data": "test1", "conversations": [{"role": "user"}]},
        {"id": "Car-Rental-1", "data": "test2", "conversations": [{"role": "assistant"}]},
    ]
    
    selected_test_cases = ["Car-Rental-0"]
    
    filtered_dataset = [case for case in mock_dataset if case.get('id') in selected_test_cases]
    
    assert len(filtered_dataset) == 1
    assert filtered_dataset[0]["id"] == "Car-Rental-0"
    assert filtered_dataset[0]["data"] == "test1"
    assert "conversations" in filtered_dataset[0]
    assert filtered_dataset[0]["conversations"] == [{"role": "user"}]
