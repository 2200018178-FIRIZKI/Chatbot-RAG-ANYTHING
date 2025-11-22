from app.query_chatbot import ask
from app.process_pdf import run_process_pdf


def main():
    print("===============================================")
    print("     🚀 RAGAnything Chatbot — APBD Sleman      ")
    print("===============================================")
    print("1) Proses PDF (wajib jika data belum di-load)")
    print("2) Langsung masuk mode chatbot")
    print("===============================================\n")

    choice = input("Pilih (1/2): ").strip()

    # --------------------------
    # 1. Proses PDF terlebih dulu
    # --------------------------
    if choice == "1":
        print("\n🔄 Memproses PDF... mohon tunggu sebentar.\n")
        try:
            run_process_pdf()
        except Exception as e:
            print("\n❌ Gagal memproses PDF:", e)
            print("Periksa file PDF dan konfigurasi.\n")
            return
        
        print("\n✅ PDF telah diproses sepenuhnya!")
        print("Sekarang kamu bisa bertanya ke chatbot 😊\n")

    # --------------------------
    # 2. Mode Chatbot
    # --------------------------
    print("===============================================")
    print("           🤖 Mode Chatbot RAGAnything         ")
    print("Ketik 'exit' atau 'quit' untuk keluar.")
    print("===============================================\n")

    while True:
        user_input = input("🧑 Kamu : ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Keluar dari chatbot. Sampai jumpa!\n")
            break

        try:
            answer = ask(user_input)
            print(f"\n🤖 Bot  : {answer}\n")
            print("-" * 60)
        except Exception as e:
            print("\n❌ Terjadi error:", str(e))
            print("Silakan cek kembali input atau konfigurasi Anda.\n")


if __name__ == "__main__":
    main()
