#!/bin/bash

echo "🔧 Installing RAGAnything Chatbot dependencies..."

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📋 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "📝 Next steps:"
echo "1. Get your OpenRouter API key from: https://openrouter.ai/"
echo "2. Update your API key in .env file"
echo "3. Test API with: python test_api.py"
echo "4. Run the chatbot: python main.py"
echo ""
echo "🚀 Happy coding!"