import asyncio
from dotenv import load_dotenv
import os

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Verificar que la variable OPENAI_API_KEY esté cargada
assert os.getenv("OPENAI_API_KEY"), "La variable de entorno OPENAI_API_KEY no está configurada"

from .manager import ResearchManager


async def main() -> None:
    query = input("What would you like to research? ")
    await ResearchManager().run(query)


if __name__ == "__main__":
    asyncio.run(main())
