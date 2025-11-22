import asyncio
from app.rag_init import get_rag


async def _aask(question: str):
    """
    Fungsi asynchronous untuk melakukan query ke RAGAnything.
    Menggunakan mode 'local' agar 100% aman token dan tidak error 402.
    """
    rag = get_rag()

    result = await rag.aquery(
        question,
        mode="local",   # 🔥 MODE PALING MURAH DAN AMAN UNTUK OPENROUTER GRATIS
    )

    # Amankan output
    if isinstance(result, str):
        return result
    return str(result)


def ask(question: str) -> str:
    """
    Fungsi synchronous untuk dipanggil dari main.py atau CLI.
    """

    try:
        return asyncio.run(_aask(question))
    except RuntimeError:
        # Jika event loop sudah berjalan (misal di lingkungan tertentu)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_aask(question))


# Testing manual
if __name__ == "__main__":
    q = "Apa isi utama dokumen APBD Sleman 2025?"
    print("Input:", q)
    print("Jawaban:", ask(q))
