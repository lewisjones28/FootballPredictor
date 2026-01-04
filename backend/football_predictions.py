import logging

# Thin wrapper to preserve CLI behavior while using the new modular structure
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from backend.models.engine import main

if __name__ == "__main__":
    main()
