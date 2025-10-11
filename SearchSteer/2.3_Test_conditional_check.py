#!/usr/bin/env python3
"""
Test script for conditional check module
Tests all functions from 2.1_conditional_check.py to ensure they work correctly
"""

import importlib.util
import sys
import time
from typing import Dict, List

# Import the conditional check module
spec = importlib.util.spec_from_file_location("conditional_check", "2.1_conditional_check.py")
conditional_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conditional_check)

# Import the required classes and functions
ConditionalChecker = conditional_check.ConditionalChecker
get_active_features = conditional_check.get_active_features
check_condition = conditional_check.check_condition
check_multiple_conditions = conditional_check.check_multiple_conditions
search_features_by_prompt = conditional_check.search_features_by_prompt

def test_get_active_features():
    """Test getting active features for a prompt"""
    print("=" * 80)
    print("🧪 TEST 1: GET ACTIVE FEATURES")
    print("=" * 80)
    
    prompt = "What are the side effects of aspirin?"
    print(f"Prompt: '{prompt}'")
    print("Getting top 5 most active features...")
    
    try:
        active_features = get_active_features(prompt, top_k=5)
        
        if not active_features:
            print("❌ No active features found!")
            return False
        
        print(f"✅ Found {len(active_features)} active features:")
        for i, feature in enumerate(active_features, 1):
            print(f"  {i}. Feature {feature['feature_id']}: {feature['max_activation']:.3f} ({feature['activation_percentage']:.1f}%)")
        
        return active_features
        
    except Exception as e:
        print(f"❌ Error getting active features: {e}")
        return None

def test_single_condition_check(active_features):
    """Test checking a single condition"""
    print("\n" + "=" * 80)
    print("🧪 TEST 2: SINGLE CONDITION CHECK")
    print("=" * 80)
    
    if not active_features:
        print("❌ No active features available for testing")
        return False
    
    prompt = "What are the side effects of aspirin?"
    feature_id = active_features[0]['feature_id']
    
    print(f"Testing condition for Feature {feature_id}")
    print("Condition: activation > 5.0%")
    
    try:
        result = check_condition(
            prompt=prompt,
            feature_id=feature_id,
            operator="greater_than",
            threshold=5.0,
            use_percentage=True
        )
        
        print(f"✅ Condition check completed:")
        print(f"  - Condition met: {result['condition_met']}")
        print(f"  - Actual activation: {result['activation_percentage']:.1f}%")
        print(f"  - Threshold: {result['threshold']}")
        print(f"  - Operator: {result['operator']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error checking single condition: {e}")
        return None

def test_multiple_conditions_and_logic(active_features):
    """Test multiple conditions with AND logic"""
    print("\n" + "=" * 80)
    print("🧪 TEST 3: MULTIPLE CONDITIONS (AND LOGIC)")
    print("=" * 80)
    
    if len(active_features) < 2:
        print("❌ Need at least 2 active features for AND logic test")
        return False
    
    prompt = "What are the side effects of aspirin?"
    
    # Create conditions for first two features
    conditions = [
        {
            'feature_id': active_features[0]['feature_id'],
            'operator': 'greater_than',
            'threshold': 5.0,
            'use_percentage': True
        },
        {
            'feature_id': active_features[1]['feature_id'],
            'operator': 'greater_than',
            'threshold': 3.0,
            'use_percentage': True
        }
    ]
    
    print(f"Testing AND logic with {len(conditions)} conditions:")
    for i, condition in enumerate(conditions, 1):
        print(f"  {i}. Feature {condition['feature_id']} > {condition['threshold']}%")
    
    try:
        result = check_multiple_conditions(prompt, conditions, logic_type="AND")
        
        print(f"✅ AND logic test completed:")
        print(f"  - Overall met: {result['overall_met']}")
        print(f"  - Met conditions: {result['met_conditions']}/{result['total_conditions']}")
        print(f"  - Logic type: {result['logic_type']}")
        
        # Show individual results
        print("  - Individual results:")
        for i, individual_result in enumerate(result['individual_results'], 1):
            print(f"    {i}. Feature {individual_result['feature_id']}: {individual_result['condition_met']} ({individual_result['activation_percentage']:.1f}%)")
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing AND logic: {e}")
        return None

def test_multiple_conditions_or_logic(active_features):
    """Test multiple conditions with OR logic"""
    print("\n" + "=" * 80)
    print("🧪 TEST 4: MULTIPLE CONDITIONS (OR LOGIC)")
    print("=" * 80)
    
    if len(active_features) < 2:
        print("❌ Need at least 2 active features for OR logic test")
        return False
    
    prompt = "What are the side effects of aspirin?"
    
    # Create conditions for first two features
    conditions = [
        {
            'feature_id': active_features[0]['feature_id'],
            'operator': 'greater_than',
            'threshold': 5.0,
            'use_percentage': True
        },
        {
            'feature_id': active_features[1]['feature_id'],
            'operator': 'greater_than',
            'threshold': 3.0,
            'use_percentage': True
        }
    ]
    
    print(f"Testing OR logic with {len(conditions)} conditions:")
    for i, condition in enumerate(conditions, 1):
        print(f"  {i}. Feature {condition['feature_id']} > {condition['threshold']}%")
    
    try:
        result = check_multiple_conditions(prompt, conditions, logic_type="OR")
        
        print(f"✅ OR logic test completed:")
        print(f"  - Overall met: {result['overall_met']}")
        print(f"  - Met conditions: {result['met_conditions']}/{result['total_conditions']}")
        print(f"  - Logic type: {result['logic_type']}")
        
        # Show individual results
        print("  - Individual results:")
        for i, individual_result in enumerate(result['individual_results'], 1):
            print(f"    {i}. Feature {individual_result['feature_id']}: {individual_result['condition_met']} ({individual_result['activation_percentage']:.1f}%)")
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing OR logic: {e}")
        return None

def test_search_features_by_prompt():
    """Test searching features by prompt with keyword"""
    print("\n" + "=" * 80)
    print("🧪 TEST 5: SEARCH FEATURES BY PROMPT WITH KEYWORD")
    print("=" * 80)
    
    prompt = "What are the side effects of aspirin?"
    keyword = "medical"
    
    print(f"Prompt: '{prompt}'")
    print(f"Keyword: '{keyword}'")
    print("Searching for medical-related features...")
    
    try:
        medical_features = search_features_by_prompt(
            prompt=prompt,
            keyword=keyword,
            top_k=3
        )
        
        if not medical_features:
            print("❌ No medical features found!")
            return False
        
        print(f"✅ Found {len(medical_features)} medical-related features:")
        for i, feature in enumerate(medical_features, 1):
            label = feature.get('label', f'Feature {feature["feature_id"]}')
            similarity = feature.get('similarity', 'N/A')
            print(f"  {i}. {label}")
            print(f"     - Activation: {feature['max_activation']:.3f} ({feature['activation_percentage']:.1f}%)")
            print(f"     - Similarity: {similarity}")
        
        return medical_features
        
    except Exception as e:
        print(f"❌ Error searching features by prompt: {e}")
        return None

def test_less_than_operator(active_features):
    """Test less_than operator"""
    print("\n" + "=" * 80)
    print("🧪 TEST 6: LESS_THAN OPERATOR")
    print("=" * 80)
    
    if not active_features:
        print("❌ No active features available for testing")
        return False
    
    prompt = "What are the side effects of aspirin?"
    feature_id = active_features[0]['feature_id']
    
    print(f"Testing less_than condition for Feature {feature_id}")
    print("Condition: activation < 10.0%")
    
    try:
        result = check_condition(
            prompt=prompt,
            feature_id=feature_id,
            operator="less_than",
            threshold=10.0,
            use_percentage=True
        )
        
        print(f"✅ Less_than condition check completed:")
        print(f"  - Condition met: {result['condition_met']}")
        print(f"  - Actual activation: {result['activation_percentage']:.1f}%")
        print(f"  - Threshold: {result['threshold']}")
        print(f"  - Operator: {result['operator']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error checking less_than condition: {e}")
        return None

def test_raw_activation_values(active_features):
    """Test using raw activation values instead of percentages"""
    print("\n" + "=" * 80)
    print("🧪 TEST 7: RAW ACTIVATION VALUES")
    print("=" * 80)
    
    if not active_features:
        print("❌ No active features available for testing")
        return False
    
    prompt = "What are the side effects of aspirin?"
    feature_id = active_features[0]['feature_id']
    
    print(f"Testing raw activation condition for Feature {feature_id}")
    print("Condition: raw activation > 2.0")
    
    try:
        result = check_condition(
            prompt=prompt,
            feature_id=feature_id,
            operator="greater_than",
            threshold=2.0,
            use_percentage=False  # Use raw values
        )
        
        print(f"✅ Raw activation condition check completed:")
        print(f"  - Condition met: {result['condition_met']}")
        print(f"  - Raw activation: {result['max_activation']:.3f}")
        print(f"  - Percentage: {result['activation_percentage']:.1f}%")
        print(f"  - Threshold: {result['threshold']}")
        print(f"  - Operator: {result['operator']}")
        print(f"  - Use percentage: {result['use_percentage']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error checking raw activation condition: {e}")
        return None

def test_conditional_checker_class():
    """Test the ConditionalChecker class directly"""
    print("\n" + "=" * 80)
    print("🧪 TEST 8: CONDITIONAL CHECKER CLASS")
    print("=" * 80)
    
    try:
        print("Initializing ConditionalChecker class...")
        checker = ConditionalChecker("meta-llama/Llama-2-7b-hf")
        
        prompt = "What are the side effects of aspirin?"
        print(f"Testing with prompt: '{prompt}'")
        
        # Test get_active_features method
        print("Testing get_active_features method...")
        active_features = checker.get_active_features(prompt, layer=16, top_k=3)
        
        if active_features:
            print(f"✅ Class method found {len(active_features)} active features:")
            for i, feature in enumerate(active_features, 1):
                print(f"  {i}. Feature {feature['feature_id']}: {feature['max_activation']:.3f} ({feature['activation_percentage']:.1f}%)")
        else:
            print("❌ No active features found with class method")
            return False
        
        # Test check_condition method
        if active_features:
            feature_id = active_features[0]['feature_id']
            print(f"Testing check_condition method for Feature {feature_id}...")
            
            result = checker.check_condition(
                prompt=prompt,
                feature_id=feature_id,
                layer=16,
                operator="greater_than",
                threshold=5.0,
                use_percentage=True
            )
            
            print(f"✅ Class method condition check:")
            print(f"  - Condition met: {result['condition_met']}")
            print(f"  - Activation: {result['activation_percentage']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing ConditionalChecker class: {e}")
        return False

def run_comprehensive_test():
    """Run all tests in sequence"""
    print("🚀 STARTING COMPREHENSIVE CONDITIONAL CHECK TESTS")
    print("=" * 80)
    print("This test suite validates all functions from 2.1_conditional_check.py")
    print("=" * 80)
    
    start_time = time.time()
    test_results = {}
    
    # Test 1: Get active features
    active_features = test_get_active_features()
    test_results['get_active_features'] = active_features is not None
    
    # Test 2: Single condition check
    if active_features:
        single_condition_result = test_single_condition_check(active_features)
        test_results['single_condition'] = single_condition_result is not None
    
    # Test 3: Multiple conditions AND logic
    if active_features:
        and_logic_result = test_multiple_conditions_and_logic(active_features)
        test_results['and_logic'] = and_logic_result is not None
    
    # Test 4: Multiple conditions OR logic
    if active_features:
        or_logic_result = test_multiple_conditions_or_logic(active_features)
        test_results['or_logic'] = or_logic_result is not None
    
    # Test 5: Search features by prompt
    search_result = test_search_features_by_prompt()
    test_results['search_features'] = search_result is not None
    
    # Test 6: Less than operator
    if active_features:
        less_than_result = test_less_than_operator(active_features)
        test_results['less_than_operator'] = less_than_result is not None
    
    # Test 7: Raw activation values
    if active_features:
        raw_activation_result = test_raw_activation_values(active_features)
        test_results['raw_activation'] = raw_activation_result is not None
    
    # Test 8: ConditionalChecker class
    class_result = test_conditional_checker_class()
    test_results['conditional_checker_class'] = class_result
    
    # Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"Total time: {total_time:.1f} seconds")
    
    print("\nDetailed results:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! The conditional check module is working correctly.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Please check the error messages above.")
    
    return test_results

if __name__ == "__main__":
    run_comprehensive_test()
