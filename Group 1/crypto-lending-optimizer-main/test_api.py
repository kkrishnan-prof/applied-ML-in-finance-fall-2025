"""
Test script for the Lending Rate API
Demonstrates all API endpoints
"""

import requests
import json
from loguru import logger

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check"""
    logger.info("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"\nHealth Check:")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def test_market_summary():
    """Test market summary"""
    logger.info("Testing market summary...")
    response = requests.get(f"{BASE_URL}/market_summary")
    print(f"\nMarket Summary:")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def test_calculate_rate():
    """Test rate calculation"""
    logger.info("Testing rate calculation...")

    scenario = {
        "asset": "BTC",
        "position": "long",
        "leverage": 10.0,
        "collateral": 1000.0
    }

    response = requests.post(f"{BASE_URL}/calculate_rate", json=scenario)
    print(f"\nRate Calculation for {scenario['leverage']}x leverage:")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def test_simulate_trade():
    """Test trade simulation"""
    logger.info("Testing trade simulation...")

    simulation = {
        "scenario": {
            "asset": "BTC",
            "position": "long",
            "leverage": 10.0,
            "collateral": 1000.0
        },
        "duration_hours": 24
    }

    response = requests.post(f"{BASE_URL}/simulate_trade", json=simulation)
    print(f"\nTrade Simulation (24h, 10x leverage):")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def test_rate_comparison():
    """Test rate comparison across leverage levels"""
    logger.info("Testing rate comparison...")

    response = requests.get(f"{BASE_URL}/rate_comparison?asset=BTC")
    print(f"\nRate Comparison Across Leverage Levels:")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def main():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("TESTING LENDING RATE API")
    logger.info("="*80 + "\n")

    try:
        # Test all endpoints
        test_health()
        test_market_summary()
        test_calculate_rate()
        test_simulate_trade()
        test_rate_comparison()

        logger.info("\n" + "="*80)
        logger.info("ALL TESTS PASSED!")
        logger.info("="*80 + "\n")

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to API. Make sure the server is running:")
        logger.error("  cd /path/to/project && PYTHONPATH=. python api/main.py")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")


if __name__ == "__main__":
    main()
