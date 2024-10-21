import pyttsx3

# Initialize the TTS
engine = pyttsx3.init()

# Set properties like voice rate and volume
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume level

# Text to speech
text = "Hello, This is a text to speech model."

# Make the engine say the text
engine.say(text)

# Wait for the speech to finish
engine.runAndWait()
