import uvicorn
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("uvicorn-launcher")


def main():
    try:
        from main import app
        logger.info("Successfully imported FastAPI app from main.py")

        try:
            from rdkit import Chem
            from rdkit.Chem import rdBase
            logger.info(f"RDKit version: {rdBase.rdkitVersion}")
        except ImportError as rdkit_err:
            from rdkit.Chem import rdBase
            logger.error("RDKit import failed! Check installation.")
            logger.exception(rdkit_err)
            sys.exit(1)

        reload = os.getenv("ENV", "development") == "development"
        workers = 1 if reload else int(os.getenv("UVICORN_WORKERS", 2))

        logger.info(f"Starting Uvicorn with {'reload' if reload else f'{workers} workers'}")

        uvicorn_config = {
            "app": "main:app",
            "host": "0.0.0.0",
            "port": 8000,
            "reload": reload,
            "log_level": "debug"
        }

        # Не используем workers в режиме reload
        if not reload:
            uvicorn_config["workers"] = workers

        uvicorn.run(**uvicorn_config)

    except ImportError as import_err:
        logger.critical("Failed to import main application!")
        logger.exception(import_err)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
    