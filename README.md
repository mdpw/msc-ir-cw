# msc-ir-cw

cd brandix-isps

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import sentence_transformers; print('✅ Setup complete!')"