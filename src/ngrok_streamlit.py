from pyngrok import ngrok
import os
import time

# Run Streamlit app on port 8502
os.system("start /B streamlit run src/frontend.py --server.port 8502")

# Wait a few seconds to make sure Streamlit starts
time.sleep(5)

# Create a public URL via ngrok
public_url = ngrok.connect(8502)
print("Public Streamlit URL:", public_url)

