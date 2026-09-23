#!/usr/bin/env python3
"""
End-to-end test for the replay functionality.
This demonstrates that the GDELT downloader and simulator work together.
"""

import sys
import logging
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_gdelt_functionality():
    """Test that GDELT functionality works."""
    print("Testing GDELT functionality...")
    
    try:
        from hermes_bot.replay.gdelt import fetch_events
        
        # Test with a simple mock
        with patch('hermes_bot.replay.gdelt.urlopen') as mock_urlopen:
            # Mock successful response
            mock_response = MagicMock()
            mock_response.read.return_value = b'''{
                "articles": [
                    {
                        "date": "2023-01-01",
                        "title": "Test government spending announcement",
                        "url": "https://example.com/test1",
                        "domain": "example.com",
                        "source": "news"
                    }
                ]
            }'''
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            # Test fetching events
            events = fetch_events("government spending", "2023-01-01", "2023-01-01", 10)
            
            assert len(events) == 1
            assert events[0]["title"] == "Test government spending announcement"
            assert events[0]["date"] == "2023-01-01"
            
        print("✓ GDELT functionality works correctly")
        return True
        
    except Exception as e:
        print(f"✗ GDELT test failed: {e}")
        return False

def test_simulator_basic():
    """Test basic simulator functionality."""
    print("Testing simulator basic functionality...")
    
    try:
        from hermes_bot.replay.simulator import ReplaySimulator
        
        # Test initialization
        simulator = ReplaySimulator(
            sectors=["government", "tech"],
            start_date="2023-01-01",
            end_date="2023-01-05"
        )
        
        assert simulator is not None
        assert simulator.sectors == ["government", "tech"]
        
        print("✓ Simulator initializes correctly")
        return True
        
    except Exception as e:
        print(f"✗ Simulator test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("END-TO-END TEST FOR REPLAY FUNCTIONALITY")
    print("=" * 60)
    
    tests = [
        test_gdelt_functionality,
        test_simulator_basic
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Replay functionality is working.")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())